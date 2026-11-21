# -*- coding: utf-8 -*-
"""
MaixCAM 延迟分解 v3（利用 nn.NN.forward_image 实现真正的分段）
==============================================================
基于探针（probe_api.txt）实测到的固件接口：

  · maix.nn.YOLO11 是 built-in 类，无 Python 级内部方法（v1 猴子补丁方案作废）
  · **maix.nn.NN 存在且可用**：
        forward(self, inputs: maix.tensor.Tensors, copy_result=True, ...)   # 只吃张量
        forward_image(...)                                                  # 吃 Image
        inputs_info() / outputs_info() / extra_info() / set_dual_buff()
  · 因此可以用"同一 cvimodel 走通用 NN"把 YOLO 解码+NMS（后处理）单独量出来。

分段口径（全部为实测，非估算）：
  ┌ t1 端到端 detect(原图)                  端到端总延迟
  ├ t2 通用 NN.forward_image(原图)          预处理 + NPU 推理
  ├ t3 端到端 detect(预缩放 320×224 图)     端到端 − resize 分量
  └ t4 通用 NN.forward_image(预缩放图)      非 resize 预处理 + NPU 推理

  后处理            = t1 − t2     （YOLO 解码 + NMS）
  resize 分量       = t1 − t3     （A/B 对照；并用 t2 − t4 互验）
  非 resize 预处理 + 推理 = t4
  纯 NPU 推理上界   = t4
  纯 NPU 推理下界   = t4 − t_fmt（t_fmt 为 Python 级 to_format 耗时，作为
                                 "格式转换+归一化+拷入张量"这部分预处理的上界代理）

同时记录 sys.npu_freq / cpu_freq / memory_info（NPU 占用率不可读的替代指标）。

用法（板子上）：
    python /root/board_latency_breakdown.py
    python /root/board_latency_breakdown.py --mud /root/models/yolo11n_random.mud --out /root/results_lat_random

输出（OUT_DIR）：
    sysapi.txt               探针信息 + 分段结果（请把这个文件发回）
    latency_breakdown.csv    逐帧：t1~t4 与派生量
    summary.txt              汇总（均值/占比）
"""
import csv
import os
import sys

from maix import image, nn, time

# ==================== 配置区 ====================
MODEL_MUD = "/root/models/yolo11n_hard.mud"
IMAGES_DIR = "/root/yolo_data_test/images/test"
OUT_DIR = "/root/results_lat"
RESIZED_DIR = "/root/test_images_320x224"
CONF_TH = 0.25
IOU_TH = 0.45
DUAL_BUFF = False      # 必须 False：真实单帧串行延迟
WARMUP = 3
MAX_FRAMES = 0         # 0 = 全部
# ==============================================

USAGE = """用法: python board_latency_breakdown.py [选项]
  --mud PATH     模型描述文件（默认 %s）
  --images DIR   测试图片目录（默认 %s）
  --out DIR      结果输出目录（默认 %s）
  --conf F       置信度阈值（默认 %s）
  --iou F        NMS IoU 阈值（默认 %s）
  --frames N     最多测多少帧（默认 0=全部）
  -h, --help     显示帮助
""" % (MODEL_MUD, IMAGES_DIR, OUT_DIR, CONF_TH, IOU_TH)

LINES = []


def out(s=""):
    """写入报告（最后统一打印一次，便于整体复制）"""
    LINES.append(str(s))


def prog(s=""):
    """运行时的即时进度输出（不进报告）"""
    print(s)


def sec(t):
    out("")
    out("=" * 64)
    out(t)
    out("=" * 64)


def us():
    try:
        return time.ticks_us()
    except Exception:
        return time.ticks_ms() * 1000


def parse_cli(argv):
    global MODEL_MUD, IMAGES_DIR, OUT_DIR, CONF_TH, IOU_TH, MAX_FRAMES
    m = {"--mud": ("MODEL_MUD", str), "--images": ("IMAGES_DIR", str),
         "--out": ("OUT_DIR", str), "--conf": ("CONF_TH", float),
         "--iou": ("IOU_TH", float), "--frames": ("MAX_FRAMES", int)}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k in ("-h", "--help"):
            print(USAGE)
            raise SystemExit(0)
        if k not in m or i + 1 >= len(argv):
            print("[错误] 参数问题: %s\n%s" % (k, USAGE))
            raise SystemExit(2)
        name, cast = m[k]
        globals()[name] = cast(argv[i + 1])
        i += 2


def safe(label, fn):
    try:
        r = fn()
        out("[OK]   %s = %r" % (label, r))
        return r
    except Exception as e:
        out("[FAIL] %s : %s: %s" % (label, type(e).__name__, e))
        return None


def stat(a):
    a = [x for x in a if isinstance(x, (int, float))]
    if not a:
        return (None, None, None)
    return (round(sum(a) / len(a), 3), round(min(a), 3), round(max(a), 3))


def list_images(d):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted(f for f in os.listdir(d) if f.lower().endswith(exts))


# ---------------------------------------------------------------- 探针
def probe(det, generic):
    sec("1. 固件与系统信息")
    try:
        from maix import sys as msys
        for fn in ("maixpy_version", "runtime_version", "os_version", "device_name",
                   "host_name", "device_id"):
            safe("sys.%s()" % fn, getattr(msys, fn))
        for fn in ("npu_freq", "cpu_freq"):
            safe("sys.%s()" % fn, getattr(msys, fn))
        for fn in ("cpu_usage", "memory_info", "cpu_temp"):
            safe("sys.%s()" % fn, getattr(msys, fn))
    except Exception as e:
        out("maix.sys 不可用: %s" % e)

    sec("2. 板端实际预处理参数（与黄金参考口径一致性证据）")
    for m in ("input_width", "input_height", "input_format", "input_size", "mean",
              "scale", "labels", "label_path"):
        f = getattr(det, m, None)
        if f is None:
            out("[缺失] det.%s" % m)
        elif callable(f):
            safe("det.%s()" % m, f)
        else:
            out("[OK]   det.%s = %r（属性）" % (m, f))
    try:
        d = getattr(type(det).detect, "__doc__", "")
        out("detect 签名: %s" % str(d).split("|")[0].strip())
    except Exception:
        pass

    sec("3. 通用推理类 nn.NN 接口（分段的关键）")
    if generic is None:
        out("nn.NN 不可用，只能退回 A/B 口径")
        return
    for m in ("forward", "forward_image"):
        f = getattr(generic, m, None)
        if f is None:
            out("[缺失] nn.NN.%s" % m)
            continue
        try:
            out("nn.NN.%s 签名: %s" % (m, str(getattr(f, "__doc__", "")).split("|")[0].strip()))
        except Exception:
            pass
    safe("nn.NN.inputs_info()", generic.inputs_info)
    safe("nn.NN.outputs_info()", generic.outputs_info)
    try:
        import maix.tensor as mt
        out("[附加] maix.tensor 成员: " + ", ".join(a for a in dir(mt) if not a.startswith("_")))
    except Exception as e:
        out("[附加] maix.tensor 不可直接导入: %s（纯推理下界估计改用 to_format 代理）" % e)


# ---------------------------------------------------------------- 主流程
def main():
    parse_cli(sys.argv[1:])
    os.makedirs(OUT_DIR, exist_ok=True)
    prog("MaixCAM 延迟分解 v3  ·  mud=%s  conf=%s  iou=%s  dual_buff=%s"
        % (MODEL_MUD, CONF_TH, IOU_TH, DUAL_BUFF))

    det = None
    for name in ("YOLO11", "YOLOv8", "YOLOv5"):
        cls = getattr(nn, name, None)
        if cls is None:
            continue
        try:
            det = cls(model=MODEL_MUD, dual_buff=DUAL_BUFF)
            prog("[加载成功] nn.%s" % name)
            break
        except Exception as e:
            prog("[加载失败] nn.%s : %s" % (name, e))
    if det is None:
        prog("模型加载失败，退出")
        return 1

    generic = None
    if hasattr(nn, "NN"):
        try:
            generic = nn.NN(MODEL_MUD, dual_buff=DUAL_BUFF)
            prog("[通用推理] nn.NN 建立成功")
        except Exception as e:
            prog("[通用推理] nn.NN 建立失败: %s" % e)

    probe(det, generic)
    fwd = getattr(generic, "forward_image", None) if generic is not None else None
    use_fwd = callable(fwd)

    files = list_images(IMAGES_DIR)
    if MAX_FRAMES:
        files = files[:MAX_FRAMES]
    w, h = det.input_width(), det.input_height()
    out("")
    out("[数据] %s 共 %d 张（预热 %d）；模型输入 %dx%d"
        % (IMAGES_DIR, len(files), WARMUP, w, h))

    # 预缩放图（用于 A/B 与 forward_image 对照）
    try:
        if not (os.path.isdir(RESIZED_DIR) and list_images(RESIZED_DIR)):
            os.makedirs(RESIZED_DIR, exist_ok=True)
            for fn in files:
                image.load(os.path.join(IMAGES_DIR, fn)).resize(w, h).save(
                    os.path.join(RESIZED_DIR, fn))
            prog("[准备] 已生成预缩放图 %d 张 → %s" % (len(files), RESIZED_DIR))
    except Exception as e:
        prog("[准备] 预缩放图生成失败: %s（A/B 与 t4 将缺失）" % e)

    rows = []
    for i, fn in enumerate(files):
        po = os.path.join(IMAGES_DIR, fn)
        ps = os.path.join(RESIZED_DIR, fn)
        try:
            oi = image.load(po)
            t0 = us(); det.detect(oi, conf_th=CONF_TH, iou_th=IOU_TH); t1 = (us() - t0) / 1000.0
            t2 = None
            if use_fwd:
                t0 = us(); fwd(oi); t2 = (us() - t0) / 1000.0
            t3 = t4 = tfmt = None
            if os.path.exists(ps):
                si = image.load(ps)
                t0 = us(); det.detect(si, conf_th=CONF_TH, iou_th=IOU_TH); t3 = (us() - t0) / 1000.0
                if use_fwd:
                    t0 = us(); fwd(si); t4 = (us() - t0) / 1000.0
                try:   # 格式转换代理（纯推理下界用）
                    t0 = us(); si.to_format(image.Format.FMT_RGB888); tfmt = (us() - t0) / 1000.0
                except Exception:
                    tfmt = None
            if i >= WARMUP:
                rows.append([fn, round(t1, 3),
                             None if t2 is None else round(t2, 3),
                             None if t3 is None else round(t3, 3),
                             None if t4 is None else round(t4, 3),
                             None if tfmt is None else round(tfmt, 3)])
        except Exception as e:
            prog("  [跳过] %s : %s: %s" % (fn, type(e).__name__, e))
        if (i + 1) % 15 == 0:
            prog("  %d/%d  %s  t1=%.2f ms" % (i + 1, len(files), fn, t1))

    with open(os.path.join(OUT_DIR, "latency_breakdown.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "t1_detect_orig_ms", "t2_forward_image_orig_ms",
                     "t3_detect_resized_ms", "t4_forward_image_resized_ms",
                     "t_to_format_proxy_ms"])
        wr.writerows(rows)

    def col(k):
        return stat([r[k] for r in rows if r[k] is not None])

    T1, T2, T3, T4, TF = col(1), col(2), col(3), col(4), col(5)
    post = stat([r[1] - r[2] for r in rows if r[2] is not None])
    resid = stat([r[1] - r[3] for r in rows if r[3] is not None])
    resid2 = stat([r[2] - r[4] for r in rows if r[2] is not None and r[4] is not None])

    # ---------- 一致性检查：避免把无意义的减法当成结果 ----------
    ok_post = post[0] is not None and post[0] > 0.2      # 后处理必须显著为正
    ok_ab = resid[0] is not None and resid[0] > 0.2
    ab_disagree = None
    if resid[0] is not None and resid2[0] is not None and resid[0] > 0.1:
        ab_disagree = abs(resid[0] - resid2[0]) / resid[0]
    util_note = None
    util = None
    if T4[0]:
        gops = 1000.0 * 1.13 / T4[0]
        util = 100.0 * gops / 700.0      # 板端实测 NPU 频点 700 MHz
        if util > 100.0:
            util_note = ("按 t4 反推的 NPU 有效算力（%.0f GOPS）超过了 700 MHz 频点下的"
                         "名义峰值（约 700 GOPS），说明 t4 并未包含全部推理耗时或 1.13 GFLOPs "
                         "的折算偏小，该利用率不作为结论使用。" % gops)
            util = None

    lines = ["", "MaixCAM 延迟分解结果（v3，实测口径）", "=" * 64,
             "模型文件        : %s" % MODEL_MUD,
             "模型输入尺寸    : %dx%d（%s）" % (w, h, det.input_format()),
             "conf / iou      : %s / %s" % (CONF_TH, IOU_TH),
             "dual_buff       : %s（串行，真实单帧延迟口径）" % DUAL_BUFF,
             "统计帧数        : %d（预热 %d）" % (len(rows), WARMUP),
             "",
             "实测原始量（均值 / 最小 / 最大，ms）：",
             "  t1 端到端 detect(原图 320×320)        %s" % (T1,),
             "  t2 NN.forward_image(原图)             %s" % (T2,),
             "  t3 端到端 detect(预缩放 320×224)      %s" % (T3,),
             "  t4 NN.forward_image(预缩放图)         %s" % (T4,),
             "  t_fmt to_format 代理（预处理上界）    %s" % (TF,),
             "",
             "派生量与一致性：",
             "  后处理候选 = t1 − t2                    %s  %s"
             % (post, "✔ 可用" if ok_post else "✘ 非正/过小，不可用"),
             "  resize 分量 = t1 − t3（A/B）           %s  %s"
             % (resid, "✔ 可用" if ok_ab else "✘ 不可用"),
             "  resize 互验 = t2 − t4                  %s" % (resid2,),
             "  非 resize 预处理 + 推理 = t4            %s" % (T4,),
             ]
    if not ok_post:
        lines += ["",
                  "  ⚠️ t2 ≥ t1 − 0.2 ms：通用 NN.forward_image 的结果拷贝开销（其 docstring 明确",
                  "     说明 copy_result 会拷贝、'not efficient'）已经抵消甚至超过了它省掉的后处理",
                  "     时间，因此**本口径无法用减法分离后处理**。论文中不得使用该差值，",
                  "     只采用 t1 / t3 / t4 的绝对值与 resize 分量。"]
    if ab_disagree is not None and ab_disagree > 0.5:
        lines += ["  ⚠️ resize 分量的两种口径差异 %.0f%%（A/B 与 t2−t4），说明 forward_image 的",
                  "     图像预处理路径与 YOLO 类可能不完全一致，论文中只引用 A/B 口径并注明。" % (100 * ab_disagree)]

    lines += ["", "可直接写入论文的量：", "  端到端单帧延迟          : %s ms" % (T1[0],)]
    if ok_ab:
        lines.append("  resize（letterbox）分量 : %.3f ms（占端到端 %.1f%%）"
                     % (resid[0], 100.0 * resid[0] / T1[0]))
    if T3[0] is not None:
        lines.append("  非 resize 部分 + 推理 + 后处理 = t3 : %.3f ms（占端到端 %.1f%%）"
                     % (T3[0], 100.0 * T3[0] / T1[0]))
    if ok_post:
        lines.append("  后处理（YOLO 解码 + NMS）: %.3f ms（占端到端 %.1f%%）"
                     % (post[0], 100.0 * post[0] / T1[0]))
    if T4[0] is not None and TF[0] is not None:
        lines.append("  纯 NPU 推理估计区间     : [%.2f, %.2f] ms（下界 = t4 − t_fmt 代理，上界 = t4）"
                     % (max(0.0, T4[0] - TF[0]), T4[0]))
    if util is not None:
        lines.append("  名义算力利用率          : 约 %.1f%%（1.13 GFLOPs ÷ t4 ÷ 实测 NPU 频点 700 MHz）"
                     % util)
    elif util_note:
        lines.append("  名义算力利用率          : 不报告 —— %s" % util_note)

    try:
        from maix import sys as msys
        lines += ["", "运行时资源快照："]
        for fn in ("npu_freq", "cpu_freq", "cpu_usage", "memory_info", "cpu_temp"):
            f = getattr(msys, fn, None)
            if f is None:
                continue
            try:
                lines.append("  sys.%s() = %r" % (fn, f()))
            except Exception as e:
                lines.append("  sys.%s() 失败: %s" % (fn, e))
    except Exception:
        pass

    lines += ["",
              "口径说明（写入论文时照抄）：",
              "  · t2/t4 使用通用 nn.NN.forward_image，同一 cvimodel、同一预处理，",
              "    与 YOLO 类的差别仅在于不含 YOLO 解码与 NMS，故 t1 − t2 即后处理；",
              "  · resize 分量为 A/B 差值（原图 320×320 vs 预缩放 320×224），",
              "    预缩放图由脚本直接拉伸生成（与 letterbox 的像素量相当，仅用于计时）；",
              "  · 纯 NPU 推理无法直接测量（forward 需 Tensors 且固件未暴露张量构造接口），",
              "    故给出区间估计并注明口径；",
              "  · NPU 占用率（%）因固件无 npu_usage 接口不可得，仅可读频点 npu_freq。",
              ]
    txt = "\n".join(LINES + lines)
    with open(os.path.join(OUT_DIR, "sysapi.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(txt)
    print("\n[完成] 结果目录: %s（请把 sysapi.txt 发回）" % OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
