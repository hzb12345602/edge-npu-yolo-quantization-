# -*- coding: utf-8 -*-
"""
板端数据分析（论文表 4-1 / 4-2 / 4-3 数据来源）
================================================
一条命令跑完全部四件事：

  1) golden  —— 用 FP32 ONNX（best.onnx）在测试集上跑"黄金参考"，
                预处理与板端完全一致：RGB、letterbox 到 224×320、pad=114、×1/255，
                conf=0.25、NMS iou=0.45。输出 golden_results.csv
  2) compare —— 板端 INT8 结果 与 黄金参考 逐框对比（IoU≥0.5 贪心匹配），
                计算坐标绝对误差(AE)、MaxAE、RMSE、超阈值占比、误检/漏检，
                并按"小目标/低光照/模糊/普通"分层。输出 matched_*.csv + quant_error_summary.txt
  3) map     —— 三组（FP32 / INT8-random / INT8-hard）在测试集上的
                P / R / mAP@0.5 / mAP@0.5:0.95（VOC 全点插值，单类）。输出 map_summary.txt
  4) metrics —— 解析板端 metrics.csv，汇总延迟 / FPS / CPU / 内存 / 温度。输出 board_metrics_summary.txt

用法（在 Windows 上、maixcam 环境里跑）：
    D:\\anaconda3\\envs\\maixcam\\python.exe board_analysis.py all
    # 或单独跑某一步
    ... board_analysis.py golden
    ... board_analysis.py compare
    ... board_analysis.py map
    ... board_analysis.py metrics

目录约定（默认值，可用参数覆盖）：
    KEY   = D:\\maixcam_exp\\model_data\\key_data
    板端困难校准结果 : KEY\\results\\board_results.csv、metrics.csv
    板端随机校准结果 : KEY\\results_random\\board_results.csv、metrics.csv
    输出目录         : KEY\\analysis\\
"""
import argparse
import csv
import json
import math
import os
import statistics as st

# ==================== 默认路径（按需改）====================
KEY = r"D:\maixcam_exp\model_data\key_data"
DEF = {
    "onnx":    r"D:\maixcam_exp\model_data\runs\train\weights\best.onnx",
    "images":  r"D:\maixcam_exp\yolo_data_test\images\test",
    "labels":  r"D:\maixcam_exp\yolo_data_test\labels\test",
    "stats":   os.path.join(KEY, "dataset_stats_test.csv"),
    "hard":    os.path.join(KEY, "results", "board_results.csv"),
    "hard_met": os.path.join(KEY, "results", "metrics.csv"),
    "rand":    os.path.join(KEY, "results_random", "board_results.csv"),
    "rand_met": os.path.join(KEY, "results_random", "metrics.csv"),
    "outdir":  os.path.join(KEY, "analysis"),
}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
# 困难样本判定阈值（与实验方案 §5 一致：小目标 <0.02、低光照 <0.30、模糊取数据集 P20）
TH_SMALL_AREA = 0.02
TH_BRIGHT = 0.30
BLUR_TH_TRAIN = 48.0  # 训练集标定得到的模糊阈值（P20），测试集分层沿用同一口径
COORD_TOL = 2.0      # 坐标误差"超阈值"门槛（像素）
CONF_TOL = 0.05      # 置信度误差门槛


# ---------------------------------------------------------------- 工具
def imread_any(path):
    import cv2
    import numpy as np
    buf = np.fromfile(path, dtype=np.uint8)      # 兼容中文路径
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_board_csv(path):
    """读板端/黄金参考统一格式 CSV → {image: [ {conf, cls, box(4)} ]}
       box = (x1, y1, x2, y2)，浮点像素坐标
    """
    d = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            img = row["image"]
            x1, y1 = float(row["x1"]), float(row["y1"])
            x2, y2 = float(row["x2"]), float(row["y2"])
            d.setdefault(img, []).append({
                "conf": float(row["conf"]),
                "cls": int(row["class_id"]),
                "box": (x1, y1, x2, y2),
            })
    return d


def read_gt_yolo(labels_dir, images_dir):
    """读 YOLO 标签 → {image: [(box, cls)]}，像素坐标（归一化 × 图像尺寸）"""
    import cv2
    import numpy as np
    gts = {}
    for fn in sorted(os.listdir(images_dir)):
        if not fn.lower().endswith(IMG_EXT):
            continue
        lp = os.path.join(labels_dir, os.path.splitext(fn)[0] + ".txt")
        im = imread_any(os.path.join(images_dir, fn))
        h, w = im.shape[:2]
        boxes = []
        if os.path.exists(lp):
            with open(lp, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.split()
                    if len(p) < 5:
                        continue
                    c, cx, cy, bw, bh = int(float(p[0])), *[float(v) for v in p[1:5]]
                    boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                                  (cx + bw / 2) * w, (cy + bh / 2) * h, c))
        gts[fn] = [(b[:4], b[4]) for b in boxes]
    return gts


def load_stats(path):
    """读 dataset_stats_*.csv → {image: dict}"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["file"]] = {
                "brightness": float(row["brightness"]),
                "sharpness": float(row["sharpness"]),
                "min_obj_area": float(row["min_obj_area"]),
            }
    return out


def classify_hard(stats, blur_th):
    """按优先级分桶：小目标 > 低光照 > 模糊 > 普通"""
    if not stats:
        return "普通"
    if stats["min_obj_area"] < TH_SMALL_AREA:
        return "小目标"
    if stats["brightness"] < TH_BRIGHT:
        return "低光照"
    if stats["sharpness"] < blur_th:
        return "模糊"
    return "普通"


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 4)


# ---------------------------------------------------------------- 1) golden
def do_golden(a):
    from ultralytics import YOLO
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "golden_results.csv")
    m = YOLO(a.onnx, task="detect")
    files = sorted(f for f in os.listdir(a.images) if f.lower().endswith(IMG_EXT))
    n_box = 0
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "img_w", "img_h", "class_id", "class_name", "conf",
                    "x", "y", "w", "h", "x1", "y1", "x2", "y2"])
        for i, fn in enumerate(files, 1):
            r = m.predict(os.path.join(a.images, fn), imgsz=(a.h, a.w),
                          conf=a.conf, iou=a.iou, verbose=False, device="cpu")[0]
            oh, ow = r.orig_shape
            b = r.boxes
            for k in range(len(b)):
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[k].tolist()]
                w.writerow([fn, ow, oh, int(b.cls[k]), m.names[int(b.cls[k])],
                            round(float(b.conf[k]), 4),
                            round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2),
                            round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
                n_box += 1
            print(f"  [{i}/{len(files)}] {fn}  目标 {len(b)}")
    print(f"[golden] 完成：{len(files)} 张图 / {n_box} 个框 → {out}")
    print(f"[golden] 预处理：RGB + letterbox({a.h}x{a.w}) + pad=114 + /255，conf={a.conf} iou={a.iou}（与板端一致）")
    return out


# ---------------------------------------------------------------- 2) compare
def match_pairs(gold, det, iou_th=0.5):
    """贪心匹配：返回 (matched[(g,d,iou)], 未匹配golden下标, 未匹配det下标)"""
    pairs = []
    for gi, g in enumerate(gold):
        for di, d in enumerate(det):
            v = iou_xyxy(g["box"], d["box"])
            if v >= iou_th:
                pairs.append((v, gi, di))
    pairs.sort(reverse=True)
    ug, ud = set(), set()
    matched = []
    for v, gi, di in pairs:
        if gi in ug or di in ud:
            continue
        ug.add(gi)
        ud.add(di)
        matched.append((gi, di, v))
    return matched, [i for i in range(len(gold)) if i not in ug], [i for i in range(len(det)) if i not in ud]


def do_compare(a):
    os.makedirs(a.outdir, exist_ok=True)
    golden = read_board_csv(a.golden)
    stats = load_stats(a.stats)
    blur_th = a.blur_th
    vals = sorted(v["sharpness"] for v in stats.values()) if stats else []
    blur_p20_test = vals[int(len(vals) * 0.20)] if vals else None
    if blur_th is None:
        # 沿用训练集标定得到的固定阈值，避免"每个数据集各调一次阈值"的口径漂移
        blur_th = BLUR_TH_TRAIN
    print(f"[compare] 模糊阈值 = {blur_th:.2f}（训练集 P20；测试集自身 P20 = "
          f"{blur_p20_test if blur_p20_test is None else round(blur_p20_test, 2)}，仅作对照）；"
          f"小目标 <{TH_SMALL_AREA}；低光照 <{TH_BRIGHT}")

    arms = [("B_random", a.rand), ("C_hard", a.hard)]
    lines = ["板端 INT8 与 FP32 黄金参考的逐框量化误差", "=" * 60,
             f"黄金参考 : {a.golden}",
             f"匹配准则 : IoU ≥ {a.iou_match}（贪心一对一）",
             f"误差口径 : 坐标为像素绝对误差 AE；超阈值 = |Δ| ≥ {COORD_TOL} px，"
             f"置信度超阈值 = |Δ| ≥ {CONF_TOL}",
             f"分层阈值 : 小目标 area<{TH_SMALL_AREA}、低光照 brightness<{TH_BRIGHT}、"
             f"模糊 sharpness<{blur_th:.1f}（优先级 小目标>低光照>模糊>普通）",
             ""]
    detail = []
    for arm, path in arms:
        det = read_board_csv(path)
        rec = {"arm": arm, "src": path, "n_img": 0, "n_gold": 0, "n_det": 0,
               "tp": 0, "fn": 0, "fp": 0, "ae": [], "conf_ae": [], "ious": [],
               "center": [], "norm": [], "conf_bias": [], "area": [], "by_type": {}}
        rows = [["image", "hard_type", "iou", "g_x1", "g_y1", "g_x2", "g_y2",
                 "d_x1", "d_y1", "d_x2", "d_y2",
                 "ae_x1", "ae_y1", "ae_x2", "ae_y2", "ae_conf", "g_conf", "d_conf",
                 "center_err", "norm_err_pct", "gt_area"]]
        for img, gobj in sorted(golden.items()):
            dobj = det.get(img, [])
            t = classify_hard(stats.get(img), blur_th)
            rec["n_img"] += 1
            rec["n_gold"] += len(gobj)
            rec["n_det"] += len(dobj)
            m, ug, ud = match_pairs(gobj, dobj, a.iou_match)
            rec["tp"] += len(m)
            rec["fn"] += len(ug)
            rec["fp"] += len(ud)
            b = rec["by_type"].setdefault(t, {"img": 0, "gold": 0, "det": 0, "tp": 0,
                                              "fn": 0, "fp": 0, "ae": [], "conf_ae": [],
                                              "ious": [], "center": [], "norm": []})
            b["img"] += 1
            b["gold"] += len(gobj)
            b["det"] += len(dobj)
            b["tp"] += len(m)
            b["fn"] += len(ug)
            b["fp"] += len(ud)
            for gi, di, v in m:
                g, d = gobj[gi], dobj[di]
                ae = [abs(g["box"][k] - d["box"][k]) for k in range(4)]
                ae_c = abs(g["conf"] - d["conf"])
                # 中心点误差 + 以目标对角线归一化的相对误差（消除目标尺度影响）
                gcx, gcy = (g["box"][0] + g["box"][2]) / 2, (g["box"][1] + g["box"][3]) / 2
                dcx, dcy = (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2
                cerr = math.hypot(gcx - dcx, gcy - dcy)
                gw, gh = g["box"][2] - g["box"][0], g["box"][3] - g["box"][1]
                diag = math.hypot(gw, gh)
                nerr = 100.0 * cerr / diag if diag > 0 else 0.0
                area = gw * gh
                rec["ae"] += ae
                rec["conf_ae"].append(ae_c)
                rec["ious"].append(v)
                rec["center"].append(cerr)
                rec["norm"].append(nerr)
                rec["conf_bias"].append(d["conf"] - g["conf"])
                rec["area"].append(area)
                b["ae"] += ae
                b["conf_ae"].append(ae_c)
                b["ious"].append(v)
                b["center"].append(cerr)
                b["norm"].append(nerr)
                rows.append([img, t, round(v, 4)] +
                            [round(x, 2) for x in g["box"]] +
                            [round(x, 2) for x in d["box"]] +
                            [round(x, 2) for x in ae] + [round(ae_c, 4),
                                                         g["conf"], d["conf"],
                                                         round(cerr, 3), round(nerr, 3),
                                                         round(area, 1)])
        # 统计
        ae = rec["ae"]
        rec["mae"] = round(sum(ae) / len(ae), 3) if ae else None
        rec["maxae"] = round(max(ae), 3) if ae else None
        rec["rmse"] = round(math.sqrt(sum(x * x for x in ae) / len(ae)), 4) if ae else None
        rec["over_tol_pct"] = round(100.0 * sum(1 for x in ae if x >= COORD_TOL) / len(ae), 2) if ae else None
        rec["conf_mae"] = round(sum(rec["conf_ae"]) / len(rec["conf_ae"]), 4) if rec["conf_ae"] else None
        rec["conf_maxae"] = round(max(rec["conf_ae"]), 4) if rec["conf_ae"] else None
        rec["conf_over_pct"] = round(100.0 * sum(1 for x in rec["conf_ae"] if x >= CONF_TOL) / len(rec["conf_ae"]), 2) if rec["conf_ae"] else None
        rec["iou_mean"] = round(sum(rec["ious"]) / len(rec["ious"]), 4) if rec["ious"] else None
        rec["iou_min"] = round(min(rec["ious"]), 4) if rec["ious"] else None
        c = rec["center"]
        rec["center_mae"] = round(sum(c) / len(c), 3) if c else None
        rec["center_max"] = round(max(c), 3) if c else None
        n = rec["norm"]
        rec["norm_mae_pct"] = round(sum(n) / len(n), 3) if n else None
        rec["conf_bias"] = round(sum(rec["conf_bias"]) / len(rec["conf_bias"]), 4) if rec["conf_bias"] else None
        rec["r_area_ae"] = pearson(rec["area"], [max(rec["ae"][i * 4:i * 4 + 4]) for i in range(len(rec["area"]))])
        rec["r_area_norm"] = pearson(rec["area"], rec["norm"])
        for tv in rec["by_type"].values():
            ae2 = tv["ae"]
            tv["mae"] = round(sum(ae2) / len(ae2), 3) if ae2 else None
            tv["maxae"] = round(max(ae2), 3) if ae2 else None
            tv["rmse"] = round(math.sqrt(sum(x * x for x in ae2) / len(ae2)), 4) if ae2 else None
            tv["over_pct"] = round(100.0 * sum(1 for x in ae2 if x >= COORD_TOL) / len(ae2), 2) if ae2 else None
            tv["conf_mae"] = round(sum(tv["conf_ae"]) / len(tv["conf_ae"]), 4) if tv["conf_ae"] else None
            tv["center_mae"] = round(sum(tv["center"]) / len(tv["center"]), 3) if tv["center"] else None
            tv["norm_mae_pct"] = round(sum(tv["norm"]) / len(tv["norm"]), 3) if tv["norm"] else None
        cp = os.path.join(a.outdir, f"matched_{arm}.csv")
        with open(cp, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
        detail.append(rec)

        lines += [f"【{arm}】 源文件 {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
                  f"  图片数 {rec['n_img']}  黄金框 {rec['n_gold']}  板端框 {rec['n_det']}  "
                  f"匹配对 {rec['tp']}  漏检 {rec['fn']}  误检 {rec['fp']}",
                  f"  坐标 MAE {rec['mae']} px   MaxAE {rec['maxae']} px   RMSE {rec['rmse']} px   "
                  f"超 {COORD_TOL}px 占比 {rec['over_tol_pct']}%",
                  f"  中心点误差 MAE {rec['center_mae']} px  Max {rec['center_max']} px   "
                  f"归一化误差 {rec['norm_mae_pct']}%（除以目标对角线）",
                  f"  置信度 MAE {rec['conf_mae']}  MaxAE {rec['conf_maxae']}  超 {CONF_TOL} 占比 {rec['conf_over_pct']}%  "
                  f"有符号偏差 {rec['conf_bias']}（正=INT8 偏高）",
                  f"  IoU 均值 {rec['iou_mean']}  最小 {rec['iou_min']}",
                  f"  相关性：目标面积 vs 最大坐标误差 r={rec['r_area_ae']}；"
                  f"目标面积 vs 归一化误差 r={rec['r_area_norm']}",
                  "  分层：" ]
        for t in ("小目标", "低光照", "模糊", "普通"):
            v = rec["by_type"].get(t)
            if not v:
                continue
            lines.append(f"    {t:4s} 图 {v['img']:2d}  匹配 {v['tp']:2d}  漏检 {v['fn']}  误检 {v['fp']}  "
                         f"MAE {v['mae']}  MaxAE {v['maxae']}  RMSE {v['rmse']}  超阈 {v['over_pct']}%  "
                         f"中心 {v['center_mae']}  归一 {v['norm_mae_pct']}%  confMAE {v['conf_mae']}")
        lines.append("")
        print(f"[compare] {arm}: MAE={rec['mae']} MaxAE={rec['maxae']} RMSE={rec['rmse']} "
              f"超阈={rec['over_tol_pct']}% 归一化={rec['norm_mae_pct']}% 中心={rec['center_mae']}px "
              f"conf偏差={rec['conf_bias']} 漏检={rec['fn']} 误检={rec['fp']}")

    txt = "\n".join(lines)
    with open(os.path.join(a.outdir, "quant_error_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(os.path.join(a.outdir, "quant_error_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"blur_th_p20": blur_th, "coord_tol": COORD_TOL, "conf_tol": CONF_TOL,
                   "iou_match": a.iou_match, "arms": detail}, f, ensure_ascii=False, indent=2)
    print("[compare] 汇总 →", os.path.join(a.outdir, "quant_error_summary.txt"))
    return detail


# ---------------------------------------------------------------- 3) mAP
def voc_ap(rec, prec):
    """VOC 全点插值 AP"""
    mrec = [0.0] + rec + [1.0]
    mpre = [0.0] + prec + [1.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap * 100.0


def ap_at(dets_all, gts, thr):
    """dets_all: [(img, conf, box)]；gts: {img: [(box, cls)]}；单类"""
    used = {k: [False] * len(v) for k, v in gts.items()}
    ngt = sum(len(v) for v in gts.values())
    dets = sorted(dets_all, key=lambda x: -x[1])
    tp, fp = [], []
    for img, conf, box in dets:
        best, bi = 0.0, -1
        for i, (gb, _) in enumerate(gts.get(img, [])):
            if used[img][i]:
                continue
            v = iou_xyxy(box, gb)
            if v > best:
                best, bi = v, i
        if best >= thr and bi >= 0:
            used[img][bi] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    if ngt == 0:
        return 0.0, 0.0, 0.0
    ctp, cfp = 0, 0
    rec, prec = [], []
    for t, f in zip(tp, fp):
        ctp += t
        cfp += f
        rec.append(ctp / ngt)
        prec.append(ctp / max(ctp + cfp, 1))
    return voc_ap(rec, prec), rec[-1] * 100 if rec else 0.0, prec[-1] * 100 if prec else 0.0


def do_map(a):
    os.makedirs(a.outdir, exist_ok=True)
    gts = read_gt_yolo(a.labels, a.images)
    arms = [("A_FP32(ONNX)", a.golden), ("B_INT8_random", a.rand), ("C_INT8_hard", a.hard)]
    thr_list = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    lines = ["测试集检测精度（单类，VOC 全点插值）", "=" * 60,
             f"测试集：{len(gts)} 张，GT 框 {sum(len(v) for v in gts.values())} 个",
             f"判定门槛：conf ≥ {a.conf}，NMS IoU {a.iou}（三组一致）", ""]
    lines.append(f"{'组别':<16}{'框数':>6}{'TP':>6}{'FP':>6}{'FN':>6}{'P(%)':>9}{'R(%)':>9}{'AP@0.5':>10}{'mAP@.5:.95':>13}")
    out_rows = []
    for name, path in arms:
        dets = []
        det = read_board_csv(path)
        for img, objs in det.items():
            for o in objs:
                dets.append((img, o["conf"], o["box"]))
        aps = []
        for thr in thr_list:
            ap, r, p = ap_at(dets, gts, thr)
            aps.append(ap)
        ap50, r50, p50 = ap_at(dets, gts, 0.5)
        # 部署工作点（conf 已固定）下的 TP/FP/FN
        used = {k: [False] * len(v) for k, v in gts.items()}
        tp = 0
        for img, conf, box in dets:
            best, bi = 0.0, -1
            for i, (gb, _) in enumerate(gts.get(img, [])):
                if used[img][i]:
                    continue
                v = iou_xyxy(box, gb)
                if v > best:
                    best, bi = v, i
            if best >= 0.5 and bi >= 0:
                used[img][bi] = True
                tp += 1
        ngt = sum(len(v) for v in gts.values())
        fp = len(dets) - tp
        fn = ngt - tp
        lines.append(f"{name:<16}{len(dets):>6}{tp:>6}{fp:>6}{fn:>6}"
                     f"{round(100*tp/max(tp+fp,1),2):>9}{round(100*tp/max(tp+fn,1),2):>9}"
                     f"{round(ap50,2):>10}{round(sum(aps)/len(aps),2):>13}")
        out_rows.append({"arm": name, "n_box": len(dets), "tp": tp, "fp": fp, "fn": fn,
                         "P": round(100 * tp / max(tp + fp, 1), 2),
                         "R": round(100 * tp / max(tp + fn, 1), 2),
                         "AP50": round(ap50, 2),
                         "mAP50_95": round(sum(aps) / len(aps), 2),
                         "AP_per_thr": [round(x, 2) for x in aps]})
    txt = "\n".join(lines)
    with open(os.path.join(a.outdir, "map_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(os.path.join(a.outdir, "map_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
    print(txt)
    print("[map] 汇总 →", os.path.join(a.outdir, "map_summary.txt"))
    return out_rows


# ---------------------------------------------------------------- 4) metrics
def do_metrics(a):
    os.makedirs(a.outdir, exist_ok=True)
    lines = ["MaixCAM 板端性能指标（来自 metrics.csv，已剔除前 3 帧预热）", "=" * 64]
    out_rows = []
    for arm, path in (("B_random", a.rand_met), ("C_hard", a.hard_met)):
        if not os.path.exists(path):
            lines.append(f"{arm}: 找不到 {path}")
            continue
        lat, fps, cpu, npu, mem, temp, nobj = [], [], [], [], [], [], []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                def num(k):
                    try:
                        return float(row[k])
                    except Exception:
                        return None
                v = num("latency_ms")
                if v is not None:
                    lat.append(v)
                v = num("fps")
                if v is not None:
                    fps.append(v)
                v = num("cpu_pct")
                if v is not None:
                    cpu.append(v)
                v = num("npu_pct")
                if v is not None:
                    npu.append(v)
                v = num("mem_used_kb")
                if v is not None:
                    mem.append(v)
                v = num("temp_c")
                if v is not None:
                    temp.append(v)
                v = num("n_objs")
                if v is not None:
                    nobj.append(v)

        def s(x):
            if not x:
                return ("—", "—", "—")
            return (round(sum(x) / len(x), 2), round(min(x), 2), round(max(x), 2))
        l, f_, c, n, m, t = s(lat), s(fps), s(cpu), s(npu), s(mem), s(temp)
        lines += [f"【{arm}】 {path}",
                  f"  单帧延迟(ms)  均值 {l[0]}  最小 {l[1]}  最大 {l[2]}",
                  f"  等效 FPS      均值 {f_[0]}  最小 {f_[1]}  最大 {f_[2]}",
                  f"  CPU 占用(%)   均值 {c[0]}  最小 {c[1]}  最大 {c[2]}",
                  f"  NPU 占用(%)   {n[0] if n[0] != '—' else '固件不支持（maix.sys 无 npu_usage）'}",
                  f"  内存占用(KB)  均值 {m[0]}（≈{round(m[0]/1024,1) if m[0] != '—' else '—'} MB）"
                  f"  最大 {m[2]}（≈{round(m[2]/1024,1) if m[2] != '—' else '—'} MB）",
                  f"  温度(℃)      均值 {t[0]}  最大 {t[2]}",
                  f"  每帧目标数    均值 {s(nobj)[0]}",
                  ""]
        # 有效算力利用率估算：FPS × 单帧算力 / NPU 峰值算力
        gflops = a.gflops
        util = None
        if isinstance(f_[0], float) and f_[0] > 0:
            util = round(100.0 * f_[0] * gflops / (a.npu_tops * 1000.0), 2)
        lines.append(f"  有效算力利用率估算：{f_[0]} FPS × {gflops} GFLOPs ÷ {a.npu_tops} TOPS"
                     f"(INT8) ≈ {util}%（仅在 NPU 无法直接读占用率时用于论文说明）")
        lines.append("")
        out_rows.append({"arm": arm, "lat_mean": l[0], "lat_min": l[1], "lat_max": l[2],
                         "fps_mean": f_[0], "cpu_mean": c[0], "mem_kb_mean": m[0],
                         "mem_mb_mean": round(m[0] / 1024, 1) if m[0] != "—" else None,
                         "mem_kb_max": m[2], "temp_mean": t[0],
                         "npu_pct": None, "npu_note": "firmware has no npu_usage",
                         "npu_util_est_pct": util})
    txt = "\n".join(lines)
    with open(os.path.join(a.outdir, "board_metrics_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(os.path.join(a.outdir, "board_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
    print(txt)
    print("[metrics] 汇总 →", os.path.join(a.outdir, "board_metrics_summary.txt"))
    return out_rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="MaixCAM 板端数据分析（论文表 4-1/4-2/4-3）")
    ap.add_argument("step", choices=["golden", "compare", "map", "metrics", "all"])
    ap.add_argument("--onnx", default=DEF["onnx"])
    ap.add_argument("--images", default=DEF["images"])
    ap.add_argument("--labels", default=DEF["labels"])
    ap.add_argument("--stats", default=DEF["stats"])
    ap.add_argument("--golden", default=os.path.join(DEF["outdir"], "golden_results.csv"))
    ap.add_argument("--hard", default=DEF["hard"])
    ap.add_argument("--rand", default=DEF["rand"])
    ap.add_argument("--hard-met", default=DEF["hard_met"])
    ap.add_argument("--rand-met", default=DEF["rand_met"])
    ap.add_argument("--outdir", default=DEF["outdir"])
    ap.add_argument("--h", type=int, default=224, help="部署输入高")
    ap.add_argument("--w", type=int, default=320, help="部署输入宽")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--iou-match", type=float, default=0.5, help="误差对比时的匹配门槛")
    ap.add_argument("--blur-th", type=float, default=None, help="模糊阈值，默认取测试集 sharpness P20")
    ap.add_argument("--gflops", type=float, default=1.13, help="部署输入下单帧算力(GFLOPs)")
    ap.add_argument("--npu-tops", type=float, default=1.0, help="SG2002 NPU INT8 峰值算力(TOPS)")
    a = ap.parse_args()

    if a.step in ("golden", "all"):
        do_golden(a)
    if a.step in ("compare", "all"):
        do_compare(a)
    if a.step in ("map", "all"):
        do_map(a)
    if a.step in ("metrics", "all"):
        do_metrics(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
