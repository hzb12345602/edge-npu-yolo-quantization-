# -*- coding: utf-8 -*-
"""
随机采样校准集脚本（对照组）
==========================
对应实验方案: MaixCAM_YOLO_量化校准优化_实验方案.md 第 5.3 节
运行环境   : Windows + Python 3.8~3.10（仅需标准库 + 文件复制）
运行时机   : 数据集上传 MaixHub 之前，与 screen_hard_samples.py 同时运行

用途       : 模拟官方默认的"随机采样校准集"策略，作为对照组 B。

用法示例:
    python random_calib.py ^
        --train-img dataset/images/train ^
        --output random_calib_images ^
        --calib-num 150
"""

import argparse
import os
import random
import shutil


def parse_args():
    p = argparse.ArgumentParser(description="随机采样校准集（对照组）")
    p.add_argument("--train-img", required=True, help="训练图像目录")
    p.add_argument("--output", default="random_calib_images", help="输出目录")
    p.add_argument("--calib-num", type=int, default=150, help="校准集目标数量")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--copy-labels", action="store_true",
                   help="同时复制对应的标注 .txt")
    return p.parse_args()


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

    n = min(args.calib_num, len(all_imgs))
    selected = sorted(random.sample(all_imgs, n))

    for name in selected:
        shutil.copy2(os.path.join(args.train_img, name),
                     os.path.join(args.output, name))
        if args.copy_labels:
            lbl = os.path.splitext(name)[0] + ".txt"
            src_lbl = os.path.join(args.train_img, lbl)
            # 标注目录与图像目录同级时直接查；否则忽略
            alt_lbl = os.path.join(
                os.path.dirname(args.train_img), "labels", lbl)
            src = src_lbl if os.path.exists(src_lbl) else (
                alt_lbl if os.path.exists(alt_lbl) else None)
            if src:
                shutil.copy2(src, os.path.join(args.output, lbl))

    with open(os.path.join(args.output, "selected.txt"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(selected))

    print(f"[完成] 随机采样 {len(selected)} 张校准图像 -> {args.output} "
          f"(seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
