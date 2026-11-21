# -*- coding: utf-8 -*-
"""
困难样本校准集筛选脚本（实验组）
================================
对应实验方案: MaixCAM_YOLO_量化校准优化_实验方案.md 第 5.2 节
运行环境   : Windows + Python 3.8~3.10 + opencv-python + numpy
            （无需任何 YOLO 训练环境）
运行时机   : 数据集上传 MaixHub 之前，在本地执行（先筛校准集，再传训练集）

筛选维度（按优先级分桶：小目标 > 低光照 > 模糊 > 普通）:
  1. 目标尺寸 : 标注框归一化面积 w*h < --small-area  (默认 0.02)
  2. 光照强度 : 灰度均值/255 < --low-bright 或 > --high-bright (默认 0.30 / 0.85)
  3. 图像清晰度: Laplacian 方差 < --blur-threshold (默认 100)

用法示例:
    python screen_hard_samples.py ^
        --train-img dataset/images/train ^
        --train-label dataset/labels/train ^
        --output hard_calib_images ^
        --calib-num 150
"""

import argparse
import os
import random
import shutil
import json

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="困难样本校准集筛选（小目标/低光照/模糊 三维度）")
    p.add_argument("--train-img", required=True, help="训练图像目录")
    p.add_argument("--train-label", required=True, help="训练标注目录（YOLO .txt）")
    p.add_argument("--output", default="hard_calib_images", help="输出目录")
    p.add_argument("--calib-num", type=int, default=150, help="校准集目标数量")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    # 困难判定阈值
    p.add_argument("--small-area", type=float, default=0.02,
                   help="归一化面积小于此值视为小目标")
    p.add_argument("--low-bright", type=float, default=0.30,
                   help="平均亮度低于此值视为低光照")
    p.add_argument("--high-bright", type=float, default=0.85,
                   help="平均亮度高于此值视为过曝")
    p.add_argument("--blur-threshold", type=float, default=100.0,
                   help="Laplacian 方差低于此值视为模糊")
    # 各维度采样比例
    p.add_argument("--ratio-small", type=float, default=0.40, help="小目标占比")
    p.add_argument("--ratio-lowlight", type=float, default=0.25, help="低光照占比")
    p.add_argument("--ratio-blur", type=float, default=0.20, help="模糊占比")
    p.add_argument("--ratio-random", type=float, default=0.15, help="随机普通占比")
    p.add_argument("--copy-labels", action="store_true",
                   help="同时复制对应的标注 .txt（便于事后分析）")
    return p.parse_args()


def imread_any(img_path):
    """兼容中文路径的读图函数
    Windows 下 cv2.imread 无法读取含非 ASCII 字符的路径，改用 np.fromfile + imdecode。
    """
    try:
        data = np.fromfile(img_path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def calc_brightness(img_path):
    """图像平均亮度（归一化到 0~1）"""
    img = imread_any(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) / 255.0


def calc_sharpness(img_path):
    """图像清晰度（Laplacian 方差）"""
    img = imread_any(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def has_small_object(label_path, small_area):
    """标注文件中是否存在小目标（归一化面积 < small_area）"""
    if not os.path.exists(label_path):
        return False
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    continue
                if w * h < small_area:
                    return True
    return False


def classify(img_name, img_path, lbl_path, args):
    """返回样本归属桶: small / lowlight / blur / normal"""
    is_small = has_small_object(lbl_path, args.small_area)
    brightness = calc_brightness(img_path)
    sharpness = calc_sharpness(img_path)
    if brightness is None or sharpness is None:
        return "normal"  # 读取失败按普通样本处理

    is_dark = brightness < args.low_bright or brightness > args.high_bright
    is_blur = sharpness < args.blur_threshold

    # 优先级：小目标 > 低光照 > 模糊 > 普通
    if is_small:
        return "small"
    if is_dark:
        return "lowlight"
    if is_blur:
        return "blur"
    return "normal"


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    all_imgs = sorted(
        f for f in os.listdir(args.train_img)
        if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not all_imgs:
        print(f"[错误] 训练图像目录为空: {args.train_img}")
        return 1

    # ---------- 分桶 ----------
    buckets = {"small": [], "lowlight": [], "blur": [], "normal": []}
    for name in all_imgs:
        img_path = os.path.join(args.train_img, name)
        lbl_path = os.path.join(args.train_label,
                                os.path.splitext(name)[0] + ".txt")
        buckets[classify(name, img_path, lbl_path, args)].append(name)

    # ---------- 按比例抽取 ----------
    ratios = {
        "small": args.ratio_small,
        "lowlight": args.ratio_lowlight,
        "blur": args.ratio_blur,
    }
    quota = {k: int(round(args.calib_num * v)) for k, v in ratios.items()}

    selected = []
    for key, num in quota.items():
        pool = buckets[key]
        if pool:
            selected += random.sample(pool, min(num, len(pool)))

    # 不足部分用普通样本补齐
    remain = args.calib_num - len(selected)
    if remain > 0 and buckets["normal"]:
        selected += random.sample(buckets["normal"], min(remain, len(buckets["normal"])))
    selected = sorted(selected)  # 保持确定性顺序

    # ---------- 复制 ----------
    for name in selected:
        shutil.copy2(os.path.join(args.train_img, name),
                     os.path.join(args.output, name))
        if args.copy_labels:
            lbl = os.path.splitext(name)[0] + ".txt"
            src_lbl = os.path.join(args.train_label, lbl)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(args.output, lbl))

    # ---------- 报告 ----------
    print(f"[完成] 筛选出 {len(selected)} 张困难样本校准图像 -> {args.output}")
    for key in ("small", "lowlight", "blur", "normal"):
        print(f"  {key:8s}: 候选 {len(buckets[key]):4d} 张, 入选 "
              f"{sum(1 for n in selected if n in buckets[key])} 张")
    print(f"  校准集目标: {args.calib_num} 张 (seed={args.seed})")

    # 清单存档（便于复现与论文记录）
    manifest = {
        "output": args.output,
        "calib_num": len(selected),
        "seed": args.seed,
        "thresholds": {
            "small_area": args.small_area,
            "low_bright": args.low_bright,
            "high_bright": args.high_bright,
            "blur_threshold": args.blur_threshold,
        },
        "ratios": dict(ratios),
        "bucket_sizes": {k: len(v) for k, v in buckets.items()},
        "selected": selected,
    }
    with open(os.path.join(args.output, "manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output, "selected.txt"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
