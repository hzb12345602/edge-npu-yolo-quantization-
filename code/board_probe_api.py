# -*- coding: utf-8 -*-
"""
MaixCAM API 探针（为延迟三段分解找可行的计时切入点）
====================================================
背景：本固件 `maix._maix.nn.YOLO11` 是 **C++ built-in 类**，Python 层没有
      `_preprocess` / `forward` / `postprocess`，也没有内层 NN 对象，
      所以"猴子补丁套计时器"的方案不可用。

本探针一次性查清以下问题，用于确定替代的分解口径：
  1) maix.nn 里到底有哪些可用的类（是否存在通用推理类 NN / Tensor 等）；
  2) 这些类是 Python 实现还是 built-in（决定能否被包装计时）；
  3) built-in 方法的 __doc__（pybind11 通常会带参数签名，能确认可用参数）；
  4) maix.sys 的 npu_freq / cpu_freq 实际返回值（能否作为 NPU 负载替代指标）；
  5) 固件版本串（maixpy_version / runtime_version / os_version，供论文环境表使用）；
  6) det 暴露的预处理参数实际取值（mean / scale / input_format / input_size，
     用于证明板端预处理与黄金参考口径一致）；
  7) 端到端 detect() 与（若存在）通用 NN.forward() 的耗时对比，
     用于判断"后处理"占端到端延迟的比例。

用法（板子上）：
    python /root/board_probe_api.py
    python /root/board_probe_api.py --mud /root/models/yolo11n_random.mud

输出：
    控制台打印 + /root/probe_api.txt（把该文件发回即可）
"""
import os
import sys

from maix import image, nn, time

MODEL_MUD = "/root/models/yolo11n_hard.mud"
IMAGES_DIR = "/root/yolo_data_test/images/test"
OUT_TXT = "/root/probe_api.txt"
CONF_TH = 0.25
IOU_TH = 0.45
WARMUP = 3
N_TIMED = 10

LINES = []


def out(s=""):
    s = str(s)
    LINES.append(s)
    print(s)


def sec(title):
    out("")
    out("=" * 62)
    out(title)
    out("=" * 62)


def try_(label, fn):
    """安全执行并打印结果"""
    try:
        r = fn()
        out("[OK]   %s = %r" % (label, r))
        return r
    except Exception as e:
        out("[FAIL] %s : %s: %s" % (label, type(e).__name__, e))
        return None


def kind_of(obj):
    try:
        import inspect
        if inspect.isclass(obj):
            mod = getattr(obj, "__module__", "?")
            builtin = "built-in/C++" if "._maix" in mod or getattr(obj, "__doc__", "") and \
                "built-in" in str(obj.__doc__) else "Python"
            return "class(%s, %s)" % (mod, builtin)
    except Exception:
        pass
    return type(obj).__name__


def describe(name, obj, max_attr=40):
    out("- %s : %s" % (name, kind_of(obj)))
    try:
        d = getattr(obj, "__doc__", None)
        if d:
            out("    __doc__: %s" % str(d).replace("\n", " | ")[:300])
    except Exception:
        pass
    try:
        attrs = [a for a in dir(obj) if not a.startswith("__")]
        out("    成员: %s" % ", ".join(attrs[:max_attr]))
    except Exception as e:
        out("    成员不可枚举: %s" % e)
    try:
        import inspect
        if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.isbuiltin(obj):
            for m in ("forward", "detect", "load", "input_size", "input_format", "mean", "scale"):
                f = getattr(obj, m, None)
                if f is None:
                    continue
                doc = getattr(f, "__doc__", None)
                if doc:
                    out("    %s.__doc__: %s" % (m, str(doc).replace("\n", " | ")[:400]))
    except Exception:
        pass


def main():
    for i, a in enumerate(sys.argv[1:]):
        if a == "--mud" and i + 1 < len(sys.argv) - 1:
            globals()["MODEL_MUD"] = sys.argv[i + 2]

    out("MaixCAM API 探针  ·  模型: %s" % MODEL_MUD)

    # ---------- 1. 固件与系统信息 ----------
    sec("1. maix.sys 信息（版本串 / 频率 / 资源）")
    try:
        from maix import sys as msys
        out("sys 成员: " + ", ".join(a for a in dir(msys) if not a.startswith("_")))
        for fn in ("maixpy_version", "runtime_version", "os_version", "device_name",
                   "device_id", "host_name"):
            try_( "sys.%s()" % fn, getattr(msys, fn))
        for fn in ("npu_freq", "cpu_freq"):
            try_( "sys.%s()" % fn, getattr(msys, fn))
        for fn in ("cpu_usage", "memory_info", "cpu_temp"):
            try_( "sys.%s()" % fn, getattr(msys, fn))
    except Exception as e:
        out("maix.sys 不可用: %s" % e)

    # ---------- 2. maix.nn 可用类 ----------
    sec("2. maix.nn 成员（找通用推理类 / 张量类）")
    try:
        names = [a for a in dir(nn) if not a.startswith("_")]
        out("nn 成员: " + ", ".join(names))
        for nm in names:
            describe(nm, getattr(nn, nm))
    except Exception as e:
        out("maix.nn 枚举失败: %s" % e)

    # ---------- 3. 加载模型，查看实例暴露的参数 ----------
    sec("3. 检测器实例暴露的参数（证明板端预处理口径）")
    det = None
    for cls_name in ("YOLO11", "YOLOv8", "YOLOv5"):
        cls = getattr(nn, cls_name, None)
        if cls is None:
            continue
        try:
            det = cls(model=MODEL_MUD, dual_buff=False)
            out("[加载成功] nn.%s（dual_buff=False）" % cls_name)
            break
        except Exception as e:
            out("[加载失败] nn.%s : %s" % (cls_name, e))
    if det is None:
        out("模型加载失败，后续跳过")
        _flush()
        return 1
    for m in ("input_width", "input_height", "input_format", "input_size",
              "mean", "scale", "labels", "label_path"):
        f = getattr(det, m, None)
        if f is None:
            out("[缺失] det.%s" % m)
            continue
        if callable(f):
            try_("det.%s()" % m, f)
        else:
            out("[OK]   det.%s = %r（属性，非方法）" % (m, f))
    describe("type(det)", type(det))

    # ---------- 4. 计时对比：端到端 vs 通用 NN.forward ----------
    sec("4. 计时对比（判断后处理占比）")
    files = sorted(f for f in os.listdir(IMAGES_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))[:WARMUP + N_TIMED]
    imgs = [image.load(os.path.join(IMAGES_DIR, f)) for f in files]
    out("测试图片 %d 张（预热 %d）" % (len(imgs), WARMUP))

    def bench(label, fn):
        ts = []
        for i, im in enumerate(imgs):
            t0 = time.ticks_us()
            fn(im)
            dt = (time.ticks_us() - t0) / 1000.0
            if i >= WARMUP:
                ts.append(dt)
        if ts:
            out("%-26s 均值 %.3f ms  最小 %.3f  最大 %.3f  (%d 帧)"
                % (label, sum(ts) / len(ts), min(ts), max(ts), len(ts)))
        else:
            out("%-26s 无有效样本" % label)
        return ts

    t_det = bench("det.detect() 端到端", lambda im: det.detect(im, conf_th=CONF_TH, iou_th=IOU_TH))

    t_nn = None
    if hasattr(nn, "NN"):
        out("")
        out("发现 nn.NN，尝试用同一 cvimodel 直接跑 forward（不含 YOLO 后处理）")
        try:
            generic = nn.NN(MODEL_MUD, dual_buff=False)
            describe("nn.NN 实例", generic)
            t_nn = bench("nn.NN.forward(img)", lambda im: generic.forward(im))
            if t_det and t_nn:
                out("")
                out("→ 后处理(含 YOLO 解码+NMS) ≈ %.3f ms（端到端 %.3f − 通用 forward %.3f）"
                    % (t_det and (sum(t_det) / len(t_det) - sum(t_nn) / len(t_nn)),
                       sum(t_det) / len(t_det), sum(t_nn) / len(t_nn)))
                out("→ 预处理 + NPU 推理 ≈ %.3f ms（通用 forward，其中 resize 段可由 A/B 对照估计）"
                    % (sum(t_nn) / len(t_nn)))
        except Exception as e:
            out("[FAIL] nn.NN 不可用: %s: %s" % (type(e).__name__, e))
    else:
        out("")
        out("未发现 nn.NN：说明本固件的推理入口只有 built-in 的 YOLO 类，")
        out("则延迟分解只能采用「端到端 + 输入尺寸 A/B 对照 + NPU 频率/名义利用率」口径。")

    # ---------- 5. resize 段代价（A/B 对照）----------
    sec("5. resize 段代价估计（原图 vs 已缩放到模型输入尺寸的图）")
    try:
        w, h = det.input_width(), det.input_height()
        small_dir = "/root/probe_resized_%dx%d" % (w, h)
        os.makedirs(small_dir, exist_ok=True)
        for f, im in zip(files, imgs):
            im.resize(w, h).save(os.path.join(small_dir, f))
        simgs = [image.load(os.path.join(small_dir, f)) for f in files]
        out("已生成预缩放图 %d 张于 %s（%dx%d）" % (len(simgs), small_dir, w, h))
        ts_small = []
        ts_orig = []
        for i, (a, b) in enumerate(zip(simgs, imgs)):
            t0 = time.ticks_us(); det.detect(a, conf_th=CONF_TH, iou_th=IOU_TH); d1 = (time.ticks_us() - t0) / 1000.0
            t0 = time.ticks_us(); det.detect(b, conf_th=CONF_TH, iou_th=IOU_TH); d2 = (time.ticks_us() - t0) / 1000.0
            if i >= WARMUP:
                ts_small.append(d1)
                ts_orig.append(d2)
        if ts_small:
            a1, a2 = sum(ts_small) / len(ts_small), sum(ts_orig) / len(ts_orig)
            out("预缩放图输入 : 均值 %.3f ms" % a1)
            out("原图输入     : 均值 %.3f ms" % a2)
            out("差值(≈resize段): %.3f ms" % (a2 - a1))
    except Exception as e:
        out("[FAIL] A/B 对照失败: %s: %s" % (type(e).__name__, e))

    _flush()
    return 0


def _flush():
    try:
        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write("\n".join(LINES) + "\n")
        print("\n[完成] 已写入 %s —— 把这个文件发回即可" % OUT_TXT)
    except Exception as e:
        print("写入 %s 失败: %s" % (OUT_TXT, e))


if __name__ == "__main__":
    raise SystemExit(main())
