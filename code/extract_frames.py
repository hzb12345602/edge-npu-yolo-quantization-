# -*- coding: utf-8 -*-
"""
视频抽帧工具（论文1 数据集采集用）
=================================
功能：
  1. 每隔 N 帧抽取候选帧
  2. 相似帧去重（灰度缩略图平均绝对差）
  3. 统计每帧亮度与清晰度，可选把"困难帧"（低光照/模糊）分离到独立目录
  4. 输出 manifest.csv 记录每帧的统计与保留/跳过原因（论文可复现性用）

运行环境：Windows + Python 3.8~3.10 + opencv-python + numpy（无需其他依赖）

用法示例：
  # 基础：每 5 帧抽 1 张，去重后输出到 frames/
  python extract_frames.py --video shot1.mp4 shot2.mp4 --output frames --interval 5

  # 同时分离困难样本池（低光照/模糊帧单独存 hard_frames/，供困难校准集使用）
  python extract_frames.py --video shot1.mp4 --output frames --hard-dir hard_frames
"""
import argparse
import csv
import os

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="视频抽帧 + 去重 + 困难帧分离")
    p.add_argument("--video", nargs="+", required=True, help="输入视频文件（可多个）")
    p.add_argument("--output", default="frames", help="正常帧输出目录")
    p.add_argument("--interval", type=int, default=5, help="每 N 帧抽取 1 帧")
    p.add_argument("--max-frames", type=int, default=0, help="最多保留帧数（0=不限）")
    p.add_argument("--dup-thresh", type=float, default=2.0,
                   help="相似帧判定阈值（缩略图平均绝对差，越小越严格）")
    p.add_argument("--thumb", type=int, default=64, help="去重缩略图边长")
    p.add_argument("--hard-dir", default="",
                   help="困难帧独立目录（留空则不分离）")
    p.add_argument("--hard-bright", type=float, default=0.30,
                   help="亮度低于此值判为低光照困难帧")
    p.add_argument("--hard-blur", type=float, default=100.0,
                   help="Laplacian 方差低于此值判为模糊困难帧")
    p.add_argument("--min-sharpness", type=float, default=0.0,
                   help="大于 0 时丢弃过糊帧（困难池仍保留）")
    p.add_argument("--prefix", default="img", help="输出文件名前缀")
    p.add_argument("--manifest", default="", help="统计清单 csv（默认写在 output 下）")
    return p.parse_args()


def brightness_of(gray):
    return float(gray.mean()) / 255.0


def sharpness_of(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def thumb_of(gray, size):
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    if args.hard_dir:
        os.makedirs(args.hard_dir, exist_ok=True)
    manifest_path = args.manifest or os.path.join(args.output, "manifest.csv")

    kept_thumbs = []          # 已保留帧的缩略图（用于去重）
    rows = []
    total_scan = total_kept = total_dup = total_hard = 0
    idx_out = 0

    for video in args.video:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"[警告] 无法打开视频: {video}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        i = -1
        print(f"处理 {video}  (fps={fps:.1f})")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            total_scan += 1
            if i % args.interval != 0:
                continue
            if args.max_frames and total_kept >= args.max_frames:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bright = brightness_of(gray)
            sharp = sharpness_of(gray)
            is_hard = (bright < args.hard_bright) or (sharp < args.hard_blur)

            # 去重：与已保留帧比较
            th = thumb_of(gray, args.thumb)
            dup = False
            for prev in kept_thumbs:
                if float(np.mean(np.abs(th - prev))) < args.dup_thresh:
                    dup = True
                    break
            if dup:
                total_dup += 1
                rows.append([os.path.basename(video), i, f"{i / fps:.2f}" if fps else "",
                             "", f"{bright:.3f}", f"{sharp:.1f}",
                             int(is_hard), 0, "duplicate"])
                continue

            # 过糊帧过滤（可选；困难池不丢）
            if args.min_sharpness > 0 and sharp < args.min_sharpness and not args.hard_dir:
                rows.append([os.path.basename(video), i, f"{i / fps:.2f}" if fps else "",
                             "", f"{bright:.3f}", f"{sharp:.1f}",
                             int(is_hard), 0, "too_blurry"])
                continue

            out_dir = args.hard_dir if (args.hard_dir and is_hard) else args.output
            name = f"{args.prefix}_{idx_out:04d}.jpg"
            out_path = os.path.join(out_dir, name)
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            kept_thumbs.append(th)
            total_kept += 1
            if is_hard:
                total_hard += 1
            idx_out += 1
            rows.append([os.path.basename(video), i, f"{i / fps:.2f}" if fps else "",
                         out_path, f"{bright:.3f}", f"{sharp:.1f}",
                         int(is_hard), 1, ""])
        cap.release()

    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_idx", "time_s", "saved_path",
                    "brightness", "sharpness", "is_hard", "kept", "skip_reason"])
        w.writerows(rows)

    print("-" * 60)
    print(f"扫描帧数: {total_scan}")
    print(f"保留帧数: {total_kept}  (其中困难帧 {total_hard})")
    print(f"去重跳过: {total_dup}")
    print(f"正常帧目录: {args.output}")
    if args.hard_dir:
        print(f"困难帧目录: {args.hard_dir}")
    print(f"统计清单: {manifest_path}")
    print("提示：划分 train/val/test 时请按视频时间段切分，避免相邻相似帧跨集造成数据泄漏。")


if __name__ == "__main__":
    main()
