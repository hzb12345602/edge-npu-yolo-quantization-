# -*- coding: utf-8 -*-
"""
合成小数据集生成器（训练链路冒烟测试用）
=====================================
用途：在真正拍数据之前，用 24 张合成图 + YOLO 标注把
      「训练 → 导出 ONNX」链路完整跑通，提前暴露环境/参数问题。

生成结构：
  <out>/
  ├── images/train/*.jpg (16)   ├── labels/train/*.txt
  ├── images/val/*.jpg   (8)    ├── labels/val/*.txt
  └── dataset.yaml

用法：
  python make_smoke_dataset.py --out D:\\maixcam_exp\\_smoke
"""
import argparse
import os

import cv2
import numpy as np

CLASSES = ["red_box", "blue_circle"]


def draw(img, cls, cx, cy, size):
    """按类别画目标，并返回 YOLO 归一化标注行"""
    h, w = img.shape[:2]
    if cls == 0:  # 红方块
        cv2.rectangle(img, (cx - size, cy - size), (cx + size, cy + size),
                      (30, 30, 220), -1)
    else:         # 蓝圆
        cv2.circle(img, (cx, cy), size, (220, 90, 30), -1)
    return f"{cls} {cx / w:.6f} {cy / h:.6f} {2 * size / w:.6f} {2 * size / h:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_smoke")
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        img_dir = os.path.join(args.out, "images", split)
        lbl_dir = os.path.join(args.out, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in range(n):
            img = np.full((args.size, args.size, 3), 200, np.uint8)
            img += rng.integers(-15, 15, img.shape, dtype=np.int16).astype(np.uint8)
            lines = []
            for _ in range(rng.integers(1, 3)):          # 每张 1~2 个目标
                cls = int(rng.integers(0, len(CLASSES)))
                s = int(rng.integers(args.size // 12, args.size // 7))
                cx = int(rng.integers(s + 5, args.size - s - 5))
                cy = int(rng.integers(s + 5, args.size - s - 5))
                lines.append(draw(img, cls, cx, cy, s))
            name = f"{split}_{i:03d}"
            cv2.imwrite(os.path.join(img_dir, name + ".jpg"), img)
            with open(os.path.join(lbl_dir, name + ".txt"), "w",
                      encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    yaml_path = os.path.join(args.out, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")

    print(f"[完成] 合成数据集: {os.path.abspath(args.out)}")
    print(f"  train {args.n_train} 张 / val {args.n_val} 张 / {len(CLASSES)} 类")
    print(f"  dataset.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
