# -*- coding: utf-8 -*-
"""
数据集分布分析（困难样本阈值标定 + 三维度可行性判断）
====================================================
输入：MaixHub 导出目录（images/ + annotations/*.xml）或 YOLO 目录（images/ + labels/*.txt）
输出：
  - 图像宽高分布
  - 亮度分布（分位数）与低光照候选数
  - 清晰度（Laplacian 方差）分布（分位数）与模糊候选数
  - 目标归一化面积分布（分位数）与小目标候选数
  - 在建议阈值下的三维度分桶统计（含"能否各凑 20 张"的判断）

用法：
  python analyze_dataset.py --src "D:\\maixcam_exp\\first_data"
  python analyze_dataset.py --src "D:\\maixcam_exp\\first_data" --out "D:\\考研资料\\_analysis.csv"
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def imread_any(p):
    try:
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--low-bright", type=float, default=0.30)
    ap.add_argument("--high-bright", type=float, default=0.85)
    ap.add_argument("--small-area", type=float, default=0.02)
    ap.add_argument("--blur", type=float, default=-1.0,
                    help="模糊阈值；<0 表示自动取清晰度的低分位数")
    ap.add_argument("--blur-pct", type=float, default=20.0, help="自动阈值用的分位数")
    ap.add_argument("--list-smallest", type=int, default=0,
                    help="按目标面积升序列出前 N 张（查看小目标是否够小）")
    return ap.parse_args()


def load_objects(src, stem, w, h, subset=None):
    """返回该图的目标归一化面积列表
    支持两种布局：
      MaixHub:  <src>/annotations/<stem>.xml
      YOLO:     <src>/labels[/subset]/<stem>.txt
    """
    cands = []
    if subset:
        cands.append(os.path.join(src, "labels", subset, stem + ".txt"))
    cands += [os.path.join(src, "annotations", stem + ".xml"),
              os.path.join(src, "labels", stem + ".txt")]
    areas = []
    for p in cands:
        if not os.path.exists(p):
            continue
        if p.endswith(".xml"):
            root = ET.parse(p).getroot()
            for obj in root.findall("object"):
                b = obj.find("bndbox")
                bw = float(b.find("xmax").text) - float(b.find("xmin").text)
                bh = float(b.find("ymax").text) - float(b.find("ymin").text)
                areas.append((bw / w) * (bh / h))
        else:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    q = line.split()
                    if len(q) >= 5:
                        areas.append(float(q[3]) * float(q[4]))
        break
    return areas


def collect_samples(src):
    """收集 (子集名, 图片路径) 列表；兼容 MaixHub 平铺与 YOLO 子目录两种布局"""
    img_root = os.path.join(src, "images")
    if not os.path.isdir(img_root):
        return None
    items = []
    subdirs = [d for d in os.listdir(img_root)
               if os.path.isdir(os.path.join(img_root, d))]
    if subdirs:                      # YOLO 布局：images/train, images/val, ...
        for sub in sorted(subdirs):
            sd = os.path.join(img_root, sub)
            for f in sorted(os.listdir(sd)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    items.append((sub, os.path.join(sd, f)))
    else:                            # MaixHub 布局：images/ 平铺
        for f in sorted(os.listdir(img_root)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append(("", os.path.join(img_root, f)))
    return items


def main():
    a = parse_args()
    items = collect_samples(a.src)
    if items is None:
        print(f"[错误] 找不到 {os.path.join(a.src, 'images')}")
        return 1
    if not items:
        print("[错误] images/ 下没有图片")
        return 1

    rows = []
    shapes = {}
    subsets = {}
    for sub, p in items:
        f = os.path.basename(p)
        img = imread_any(p)
        if img is None:
            print(f"  [读取失败] {f}")
            continue
        h, w = img.shape[:2]
        shapes[(w, h)] = shapes.get((w, h), 0) + 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bright = float(gray.mean()) / 255.0
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        areas = load_objects(a.src, os.path.splitext(f)[0], w, h,
                             subset=sub or None)
        min_area = min(areas) if areas else -1
        subsets.setdefault(sub, 0)
        subsets[sub] += 1
        rows.append([f, w, h, round(bright, 4), round(sharp, 2),
                     len(areas), round(min_area, 5)])

    if not rows:
        print("[错误] 没有成功读取的图片")
        return 1

    br = np.array([r[3] for r in rows])
    sh = np.array([r[4] for r in rows])
    ar = np.array([r[6] for r in rows if r[6] > 0])

    print("=" * 64)
    print(f"样本数: {len(rows)}")
    if len(subsets) > 1 or list(subsets)[0]:
        print("子集: " + ", ".join(f"{k or 'images'}={v}" for k, v in subsets.items()))
        print("提示：如需按子集分别统计分布，请对 train / val / test 目录分别运行本脚本。")
    print(f"图像尺寸分布: " + ", ".join(f"{k[0]}x{k[1]}×{v}" for k, v in shapes.items()))
    print("-" * 64)

    def line(name, arr, unit=""):
        if len(arr) == 0:
            print(f"{name}: 无数据")
            return
        qs = np.percentile(arr, [0, 10, 20, 25, 50, 75, 90, 100])
        print(f"{name}: min {qs[0]:.3f} | P10 {qs[1]:.3f} | P20 {qs[2]:.3f} | "
              f"P25 {qs[3]:.3f} | 中位 {qs[4]:.3f} | P75 {qs[5]:.3f} | "
              f"P90 {qs[6]:.3f} | max {qs[7]:.3f}{unit}")

    line("平均亮度  ", br)
    line("清晰度(Lap)", sh)
    line("最小目标面积", ar)

    blur_thr = a.blur if a.blur >= 0 else float(np.percentile(sh, a.blur_pct))
    print("-" * 64)
    print(f"采用阈值：低光照 <{a.low_bright} 或 >{a.high_bright}；"
          f"模糊 <{blur_thr:.1f}（{'自动 P%.0f' % a.blur_pct if a.blur < 0 else '手动'}）；"
          f"小目标面积 <{a.small_area}")

    b_small = b_low = b_blur = b_norm = 0
    raw_small = raw_low = raw_blur = 0
    for r in rows:
        _, _, _, bright, sharp, n, min_area = r
        is_small = (min_area > 0 and min_area < a.small_area)
        is_dark = (bright < a.low_bright) or (bright > a.high_bright)
        is_blur = sharp < blur_thr
        if is_small:
            raw_small += 1
        if is_dark:
            raw_low += 1
        if is_blur:
            raw_blur += 1
        # 优先级：小目标 > 低光照 > 模糊 > 普通（与筛选脚本一致）
        if is_small:
            b_small += 1
        elif is_dark:
            b_low += 1
        elif is_blur:
            b_blur += 1
        else:
            b_norm += 1

    print("-" * 64)
    print("① 各判据原始计数（可能重叠）：")
    print(f"  满足小目标 (面积<{a.small_area}) : {raw_small:3d} 张")
    print(f"  满足低光照 (亮度<{a.low_bright} 或 >{a.high_bright}) : {raw_low:3d} 张")
    print(f"  满足模糊 (Lap<{blur_thr:.0f}) : {raw_blur:3d} 张")
    print("② 按优先级分桶（小目标 > 低光照 > 模糊 > 普通，不重叠）：")
    print(f"  小目标 : {b_small:3d} 张")
    print(f"  低光照 : {b_low:3d} 张")
    print(f"  模糊   : {b_blur:3d} 张")
    print(f"  普通   : {b_norm:3d} 张")
    need = 20
    ok = all(x >= need for x in (b_small, b_low, b_blur))
    print(f"  三维度各需 {need} 张 → {'✅ 满足' if ok else '⚠️ 不足，需调整阈值或补拍'}")
    if not ok:
        print(f"    缺口：小目标 {max(0, need - b_small)} / 低光照 {max(0, need - b_low)}"
              f" / 模糊 {max(0, need - b_blur)}")
    print("=" * 64)

    if a.out:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "width", "height", "brightness", "sharpness",
                        "n_objects", "min_obj_area"])
            w.writerows(rows)
        print(f"明细已保存: {a.out}")

    if a.list_smallest > 0:
        valid = [r for r in rows if r[6] > 0]
        valid.sort(key=lambda r: r[6])
        print("-" * 64)
        print(f"目标面积最小的 {a.list_smallest} 张（面积阈值 {a.small_area}）：")
        print(f"{'文件':<32}{'面积':>10}{'亮度':>9}{'清晰度':>10}{'是否达标':>10}")
        for r in valid[:a.list_smallest]:
            flag = "✅ 小目标" if r[6] < a.small_area else ""
            print(f"{r[0]:<32}{r[6]:>10.4f}{r[3]:>9.3f}{r[4]:>10.1f}{flag:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
