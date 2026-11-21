# -*- coding: utf-8 -*-
"""
论文关键数据汇总工具（生成 key_data 记录）
========================================
从训练日志 / results.csv / 校准集清单 / 数据集分布中提取论文所需数据，
统一写入 key_data 目录，便于写论文时直接查表。

用法：
  python collect_key_data.py \
      --run-dir   D:\\maixcam_exp\\model_data\\runs\\train \
      --key-data  D:\\maixcam_exp\\model_data\\key_data \
      --calib-dir D:\\maixcam_exp\\model_data\\calib \
      --dataset   D:\\maixcam_exp
"""
import argparse
import csv
import os
import re
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="训练输出目录（含 results.csv 与 weights/）")
    ap.add_argument("--key-data", required=True, help="key_data 输出目录")
    ap.add_argument("--calib-dir", default="", help="校准集根目录（含 small/lowlight/blur/mixed/random）")
    ap.add_argument("--dataset", default="", help="数据集根目录（含 yolo_data_train/val/test）")
    return ap.parse_args()


def read_results(run_dir):
    """读取 results.csv，返回最后一轮的指标字典 + 最优 mAP 轮次"""
    p = os.path.join(run_dir, "results.csv")
    if not os.path.exists(p):
        return None, None
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    clean = [{k.strip(): v.strip() for k, v in r.items()} for r in rows]
    last = clean[-1]
    best = max(clean, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0) or 0))
    return last, best


def model_meta_from_weights(weights):
    """直接从 .pt 计算 参数量与各输入尺寸下的 GFLOPs（比解析日志可靠）"""
    meta = {}
    try:
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import get_flops
        m = YOLO(weights)
        meta["params"] = f"{sum(p.numel() for p in m.model.parameters()):,}"
        meta["gflops_640"] = f"{get_flops(m.model, 640):.2f}"
        g320 = get_flops(m.model, 320)
        meta["gflops_320"] = f"{g320:.2f}"
        # 224×320 输入：get_flops 对非方形支持不稳定，按面积比例折算并在说明中标注
        meta["gflops_224x320"] = f"{g320 * (224 * 320) / (320 * 320):.2f}（按面积折算）"
    except Exception as e:
        meta["error"] = str(e)
    return meta


def read_log_meta(log_path):
    """从训练日志中提取推理耗时（日志含 ANSI/覆盖写入干扰，仅作兜底）"""
    meta = {}
    if not os.path.exists(log_path):
        return meta
    txt = open(log_path, encoding="utf-8", errors="ignore").read()
    txt = re.sub(r"\x1b\][^\x07]*\x07", "", txt)          # OSC
    txt = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", txt)      # CSI（含 ESC[K）
    txt = txt.replace("\r", " ")
    m = re.search(r"(\d+)\s+\S*poch\S*\s+compl\S*\s+in\s+([\d.]+)\s+hours", txt)
    if m:
        meta["epochs_done"] = m.group(1)
        meta["hours"] = m.group(2)
    # 推理耗时（日志被终端覆盖写入干扰，字母可能缺失，用宽松匹配兜底）
    m = re.search(r"([\d.]+)ms\s+pr\S*proc\S*ss,\s*([\d.]+)ms\s+inf\S*,\s*[\d.]+ms\s+loss,\s*([\d.]+)ms\s+postproc\S*ss", txt)
    if not m:
        m = re.search(r"([\d.]+)ms\s+\S+,\s*([\d.]+)ms\s+\S+,\s*([\d.]+)ms\s+\S+,\s*([\d.]+)ms\s+\S+", txt)
        if m:
            meta["preprocess_ms"] = m.group(1)
            meta["inference_ms"] = m.group(2)
            meta["postprocess_ms"] = m.group(4)
            return meta
    if m:
        meta["preprocess_ms"] = m.group(1)
        meta["inference_ms"] = m.group(2)
        meta["postprocess_ms"] = m.group(3)
    return meta


def collect_calib(calib_dir, key_data):
    """汇总四个校准集的文件清单"""
    out = []
    if not calib_dir or not os.path.isdir(calib_dir):
        return out
    for g in ("small", "lowlight", "blur", "mixed", "random"):
        d = os.path.join(calib_dir, g)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png")))
        out.append(f"[{g}] 共 {len(files)} 张")
        out.extend("    " + f for f in files)
    txt = "\n".join(out)
    with open(os.path.join(key_data, "calib_manifest.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return out


def main():
    a = parse_args()
    os.makedirs(a.key_data, exist_ok=True)

    last, best = read_results(a.run_dir)
    meta = read_log_meta(os.path.join(a.key_data, "train_log.txt"))
    weights = os.path.join(a.run_dir, "weights", "best.pt")
    if os.path.exists(weights):
        meta.update(model_meta_from_weights(weights))
    if last is not None:
        meta.setdefault("epochs_done", str(len(
            [r for r in csv.DictReader(open(os.path.join(a.run_dir, "results.csv"), encoding="utf-8"))])))

    # 1) 复制 results.csv
    src_csv = os.path.join(a.run_dir, "results.csv")
    if os.path.exists(src_csv):
        shutil.copy2(src_csv, os.path.join(a.key_data, "results.csv"))

    # 2) 校准集清单
    calib_lines = collect_calib(a.calib_dir, a.key_data)

    # 3) 权重文件大小
    wdir = os.path.join(a.run_dir, "weights")
    sizes = {}
    for n in ("best.pt", "last.pt", "best.onnx"):
        p = os.path.join(wdir, n)
        if os.path.exists(p):
            sizes[n] = round(os.path.getsize(p) / 1024 / 1024, 2)

    # 4) 写 summary.md
    def g(d, k, default="—"):
        return d.get(k, default) if d else default

    lines = []
    lines.append("# 论文关键数据汇总（key_data）\n")
    lines.append(f"生成时间：自动生成 · 训练目录：`{a.run_dir}`\n")
    lines.append("## 一、训练结果（FP32 模型）\n")
    lines.append("| 指标 | 数值 | 说明 |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| 模型 | YOLO11-n | — |")
    lines.append(f"| 参数量 | {g(meta,'params')} | 由 best.pt 计算（部署时 BN 融合后约 2,582,347） |")
    lines.append(f"| GFLOPs @320×320 | {g(meta,'gflops_320')} | 本文训练/推理分辨率 |")
    lines.append(f"| GFLOPs @224×320 | {g(meta,'gflops_224x320')} | **板端部署输入尺寸** |")
    lines.append(f"| GFLOPs @640×640 | {g(meta,'gflops_640')} | 官方基准口径（便于与公开数据对照） |")
    lines.append(f"| 训练轮数 | {g(meta,'epochs_done')} | 实际完成（计划 150） |")
    lines.append(f"| 训练耗时 | {g(meta,'hours')} 小时 | CPU 训练 |")
    lines.append(f"| **mAP@0.5**（末轮 val） | **{g(last,'metrics/mAP50(B)')}** | 论文表 4-1 A 组 |")
    lines.append(f"| **mAP@0.5:0.95**（末轮 val） | **{g(last,'metrics/mAP50-95(B)')}** | 论文表 4-1 A 组 |")
    lines.append(f"| Precision / Recall（末轮） | {g(last,'metrics/precision(B)')} / {g(last,'metrics/recall(B)')} | — |")
    lines.append(f"| 最优轮 mAP@0.5:0.95 | {g(best,'metrics/mAP50-95(B)')}（第 {g(best,'epoch')} 轮） | best.pt 依据 |")
    lines.append(f"| 推理耗时（验证时） | 预处理 {g(meta,'preprocess_ms')}ms + 推理 {g(meta,'inference_ms')}ms + 后处理 {g(meta,'postprocess_ms')}ms | CPU（非板端） |")
    lines.append("")
    lines.append("## 二、模型文件\n")
    lines.append("| 文件 | 大小(MB) |")
    lines.append("| --- | --- |")
    for k, v in sizes.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## 三、校准集构成\n")
    lines.append("```")
    lines.extend(calib_lines if calib_lines else ["（未收集）"])
    lines.append("```")
    lines.append("")
    lines.append("## 四、待板端实测回填（表 4-1 / 4-2 / 4-3）\n")
    lines.append("| 组别 | mAP@0.5 | mAP@0.5:0.95 | FPS | 延迟(ms) | 模型大小(MB) | NPU占用(%) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    lines.append("| A FP32 | 见上表（val） | 见上表 | 板端 | 板端 | — | — |")
    lines.append("| B 随机INT8 | 板端实测 | 板端实测 | 板端 | 板端 | 板端 | 板端 |")
    lines.append("| C 困难样本INT8 | 板端实测 | 板端实测 | 板端 | 板端 | 板端 | 板端 |")
    lines.append("")
    lines.append("> 板端数据来源：`metrics.csv`（FPS/延迟/CPU/NPU/内存/温度）与 `board_results.csv`（检测框）。")

    with open(os.path.join(a.key_data, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("=" * 60)
    print(f"key_data 已更新: {a.key_data}")
    print(f"  summary.md / results.csv / calib_manifest.txt")
    print(f"  末轮 mAP50={g(last,'metrics/mAP50(B)')}  mAP50-95={g(last,'metrics/mAP50-95(B)')}")
    print(f"  参数量={g(meta,'params')}  GFLOPs={g(meta,'gflops')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
