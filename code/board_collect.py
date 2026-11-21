# -*- coding: utf-8 -*-
"""
MaixCAM 板端数据采集脚本（论文 9.4 节配套）
==========================================
用途：在 MaixCAM 上对**测试集图片**逐张推理，采集两类论文数据：
  ① board_results.csv —— 每张图的检测框（类别/置信度/坐标），用于与黄金参考对比算量化误差
  ② metrics.csv       —— 每帧性能数据（延迟/FPS/CPU/NPU/内存/温度），用于论文表 4-1
  ③ summary.txt       —— 汇总统计（均值/极值），可直接抄进论文

使用步骤：
  1) 把本文件、模型（cvimodel + mud）、测试图片目录一起传到板子：
       scp board_collect.py yolo11n_hard.mud yolo11n_hard_int8.cvimodel root@<板IP>:/root/models/
       scp -r test_images root@<板IP>:/root/
  2) 修改下面的配置（至少改 MODEL_MUD 与 IMAGES_DIR）
  3) 板子上运行： python /root/board_collect.py
  4) 把 /root/results/ 下的 3 个文件拉回电脑，放进 D:\\maixcam_exp\\model_data\\key_data
       scp root@<板IP>:/root/results/* ./key_data/

⚠️ 重要（延迟测量口径）：
  DUAL_BUFF=False 时测的是**真实单帧延迟**（推荐用于论文表 4-1 的"延迟"列）；
  DUAL_BUFF=True（默认加速模式）会流水线并行，单帧延迟不可比，只适合测吞吐 FPS。
"""
import csv
import os
import sys

from maix import image, nn, time, sys as msys

# ==================== 配置区（按需修改）====================
MODEL_MUD  = "/root/models/yolo11n_random.mud"   # 模型描述文件（困难校准版 / 随机校准版）
IMAGES_DIR = "/root/yolo_data_test/images/test"               # 测试集图片目录（板子上的路径）
OUT_DIR    = "/root/results_random_0.05"                   # 结果输出目录
CONF_TH    = 0.05                              # 必须与 mud、黄金参考一致。0.05再测试一次
IOU_TH     = 0.45                              # 必须与 mud、黄金参考一致
DUAL_BUFF  = False                             # False=测真实单帧延迟；True=吞吐优先
WARMUP     = 3                                 # 前 N 张不计入统计（预热）
REPEAT     = 1                                 # 整个测试集重复遍数（想多采样可设 2~3）
USE_CAMERA = False                             # True=改用摄像头实时采集（测真实 FPS）
CAM_FRAMES = 300                               # 摄像头模式采集帧数
# ==========================================================


def list_sys_apis():
    """列出当前固件 maix.sys 实际暴露的接口，便于适配不同版本"""
    names = [n for n in dir(msys) if not n.startswith("_")]
    print("[maix.sys 可用接口] " + ", ".join(names))
    return names


def sys_metric(fn_name, key, default=-1):
    """按名字取系统指标：接口不存在/返回非 dict/抛异常 都直接返回 default。
       关键点：先 getattr 判空再调用，绝不在 try 外做属性查找。
    """
    fn = getattr(msys, fn_name, None)
    if fn is None:
        return default
    try:
        d = fn()
    except Exception:
        return default
    if not isinstance(d, dict):
        return default
    v = d.get(key, default)
    return default if v is None else v


def safe_sys_metrics():
    """读取系统指标，接口缺失/不支持时全部降级为占位值，绝不中断采集"""
    mem = sys_metric("memory_info", "used", 0)
    try:
        mem_kb = int(mem) // 1024
    except Exception:
        mem_kb = 0
    return {
        "cpu_pct": sys_metric("cpu_usage", "cpu", -1),
        # 部分固件无 npu_usage，缺失记 -1，论文中标注为“固件不支持”
        "npu_pct": sys_metric("npu_usage", "npu", -1),
        "mem_used_kb": mem_kb,
        "temp_c": sys_metric("cpu_temp", "cpu", -1),
    }


def load_detector(mud_path, dual_buff):
    """按模型类型选择正确的 MaixPy API 类：
         YOLO11 → nn.YOLO11 ； YOLOv8 / YOLOv5u → nn.YOLOv8 ； 旧版 YOLOv5 → nn.YOLOv5
       依次尝试，避免"用错类"导致加载失败或后处理异常。
    """
    for name in ("YOLO11", "YOLOv8", "YOLOv5"):
        cls = getattr(nn, name, None)
        if cls is None:
            continue
        try:
            det = cls(model=mud_path, dual_buff=dual_buff)
            print(f"[加载成功] 使用 nn.{name}")
            return det
        except Exception as e:
            print(f"[尝试 nn.{name} 失败] {e}")
    raise RuntimeError("模型加载失败：请检查 mud 是否为 INI 格式、cvimodel 是否与 mud 同目录")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[加载模型] {MODEL_MUD}")
    det = load_detector(MODEL_MUD, DUAL_BUFF)
    print(f"  输入: {det.input_width()}x{det.input_height()}  类别: {det.labels}")

    # 启动前先探明固件支持哪些系统指标：缺失的接口后续写 -1，不中断采集
    apis = list_sys_apis()
    missing = [n for n, k in (("cpu_usage", "cpu_pct"), ("npu_usage", "npu_pct"),
                              ("memory_info", "mem_used_kb"), ("cpu_temp", "temp_c"))
               if n not in apis]
    if missing:
        print("[提示] 本固件不支持这些指标，对应列将记为 -1：" + ", ".join(missing))
        if "npu_usage" in missing:
            print("[提示] 无 npu_usage 时，NPU 占用率改用『有效算力利用率』替代："
                  "1 TOPS(INT8) × 实测FPS ÷ 单帧理论算力需求，或直接以板端实测延迟/FPS 表述。")

    f_det = open(os.path.join(OUT_DIR, "board_results.csv"), "w", newline="")
    w_det = csv.writer(f_det)
    w_det.writerow(["image", "img_w", "img_h", "class_id", "class_name", "conf",
                    "x", "y", "w", "h", "x1", "y1", "x2", "y2"])

    f_met = open(os.path.join(OUT_DIR, "metrics.csv"), "w", newline="")
    w_met = csv.writer(f_met)
    w_met.writerow(["seq", "image", "latency_ms", "fps", "cpu_pct", "npu_pct",
                    "mem_used_kb", "temp_c", "n_objs"])

    lat_list, fps_list = [], []
    cpu_l, npu_l, mem_l, temp_l, obj_l = [], [], [], [], []
    seq = 0
    total_imgs = 0

    if USE_CAMERA:
        from maix import camera, display
        cam = camera.Camera(det.input_width(), det.input_height(), det.input_format())
        disp = display.Display()
        print(f"[摄像头模式] 采集 {CAM_FRAMES} 帧（按板子上的退出键可提前结束）")
        while seq < CAM_FRAMES:
            img = cam.read()
            t0 = time.ticks_ms()
            objs = det.detect(img, conf_th=CONF_TH, iou_th=IOU_TH)
            t1 = time.ticks_ms()
            lat = t1 - t0
            seq += 1
            total_imgs += 1
            if seq > WARMUP:
                lat_list.append(lat)
            m = safe_sys_metrics()
            w_met.writerow([seq, "camera", lat, round(1000.0 / lat, 2) if lat else 0,
                            m["cpu_pct"], m["npu_pct"], m["mem_used_kb"], m["temp_c"], len(objs)])
            img.draw_string(2, 2, f"{lat}ms {1000.0/max(lat,1):.1f}FPS", image.COLOR_RED)
            disp.show(img)
            if seq % 30 == 0:
                print(f"  {seq}/{CAM_FRAMES}  延迟 {lat}ms")
    else:
        files = sorted(f for f in os.listdir(IMAGES_DIR)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not files:
            print(f"[错误] {IMAGES_DIR} 下没有图片")
            return 1
        print(f"[图片模式] {len(files)} 张 × {REPEAT} 遍，预热 {WARMUP} 张")
        for r in range(REPEAT):
            for fn in files:
                p = os.path.join(IMAGES_DIR, fn)
                img = image.load(p)
                if img is None:
                    print(f"  [跳过] 读取失败: {fn}")
                    continue
                t0 = time.ticks_ms()
                objs = det.detect(img, conf_th=CONF_TH, iou_th=IOU_TH)
                t1 = time.ticks_ms()
                lat = t1 - t0
                seq += 1
                total_imgs += 1

                for o in objs:
                    w_det.writerow([fn, img.width(), img.height(),
                                    o.class_id, det.labels[o.class_id],
                                    round(o.score, 4),
                                    o.x, o.y, o.w, o.h,
                                    o.x, o.y, o.x + o.w, o.y + o.h])

                if seq > WARMUP:
                    m = safe_sys_metrics()
                    lat_list.append(lat)
                    fps_list.append(round(1000.0 / lat, 2) if lat else 0)
                    cpu_l.append(m["cpu_pct"]); npu_l.append(m["npu_pct"])
                    mem_l.append(m["mem_used_kb"]); temp_l.append(m["temp_c"])
                    obj_l.append(len(objs))
                    w_met.writerow([seq, fn, lat, fps_list[-1], m["cpu_pct"],
                                    m["npu_pct"], m["mem_used_kb"], m["temp_c"], len(objs)])
                if seq % 10 == 0:
                    print(f"  {seq}/{len(files) * REPEAT}  {fn}  延迟 {lat}ms  目标 {len(objs)}")

    f_det.close()
    f_met.close()

    def stat(a):
        a = [x for x in a if isinstance(x, (int, float)) and x >= 0]
        return (round(sum(a) / len(a), 2), min(a), max(a)) if a else ("固件不支持", "—", "—")

    avg_lat, min_lat, max_lat = stat(lat_list)
    avg_fps = round(1000.0 / avg_lat, 2) if isinstance(avg_lat, float) and avg_lat else "—"
    lines = [
        "MaixCAM 板端采集汇总",
        "=" * 44,
        f"模型文件      : {MODEL_MUD}",
        f"输入尺寸      : {det.input_width()}x{det.input_height()}",
        f"类别          : {det.labels}",
        f"conf / iou    : {CONF_TH} / {IOU_TH}",
        f"dual_buff     : {DUAL_BUFF}（False=真实单帧延迟口径）",
        f"统计样本数    : {len(lat_list)}（总推理 {total_imgs}，预热 {WARMUP}）",
        "",
        f"单帧延迟 (ms) : 均值 {avg_lat}  最小 {min_lat}  最大 {max_lat}",
        f"等效 FPS      : {avg_fps}",
        f"CPU 占用 (%)  : 均值 {stat(cpu_l)[0]}  最大 {stat(cpu_l)[2]}",
        f"NPU 占用 (%)  : 均值 {stat(npu_l)[0]}  最大 {stat(npu_l)[2]}",
        f"内存占用 (KB) : 均值 {stat(mem_l)[0]}  最大 {stat(mem_l)[2]}",
        f"温度 (℃)     : 均值 {stat(temp_l)[0]}  最大 {stat(temp_l)[2]}",
        f"每帧目标数    : 均值 {stat(obj_l)[0]}",
    ]
    txt = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    print(f"\n[完成] 结果目录: {OUT_DIR}")
    print("  请把 board_results.csv / metrics.csv / summary.txt 拉回电脑的 key_data 目录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
