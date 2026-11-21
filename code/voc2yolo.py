# -*- coding: utf-8 -*-
"""
MaixHub(VOC XML) → YOLO 格式转换工具
===================================
MaixHub 标注后导出的数据集结构为：
    <数据集>/
    ├── images/          # 图片
    ├── annotations/     # VOC 格式 XML 标注（含 bndbox）
    └── train.txt        # 文件清单

本工具把它转成本项目需要的 YOLO 结构：
    <输出>/
    ├── images/train|val/     # 图片
    ├── labels/train|val/     # YOLO txt 标注（归一化 cx cy w h）
    ├── classes.txt
    └── dataset.yaml

用法示例：
  # 全部作为验证集
  python voc2yolo.py --src "D:\\maixcam_exp\\嵌入式算法验证集" --out "D:\\maixcam_exp\\dataset" --assign val

  # 全部作为训练集
  python voc2yolo.py --src "...\\训练集" --out "D:\\maixcam_exp\\dataset" --assign train

  # 追加到已有数据集（第二次运行时用，不会覆盖已有文件）
  python voc2yolo.py --src "...\\测试集" --out "D:\\maixcam_exp\\dataset" --assign val --append
"""
import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="MaixHub VOC XML → YOLO txt 转换")
    p.add_argument("--src", required=True, help="MaixHub 导出的数据集目录（含 images/ 与 annotations/）")
    p.add_argument("--out", required=True, help="输出数据集目录（将生成 images/ labels/ dataset.yaml）")
    p.add_argument("--assign", choices=["train", "val", "test"], default="val",
                   help="这批数据归入哪个子集（默认 val）")
    p.add_argument("--append", action="store_true", help="追加模式（不清空输出目录）")
    p.add_argument("--classes", default="", help="类别名（逗号分隔）；留空则自动从 XML 收集")
    return p.parse_args()


def read_xml(path):
    """解析 VOC XML，返回 (宽, 高, [(类名, xmin, ymin, xmax, ymax), ...])"""
    root = ET.parse(path).getroot()
    size = root.find("size")
    w = int(float(size.find("width").text))
    h = int(float(size.find("height").text))
    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        b = obj.find("bndbox")
        x1 = float(b.find("xmin").text)
        y1 = float(b.find("ymin").text)
        x2 = float(b.find("xmax").text)
        y2 = float(b.find("ymax").text)
        objs.append((name, x1, y1, x2, y2))
    return w, h, objs


def main():
    args = parse_args()
    src_img = os.path.join(args.src, "images")
    src_ann = os.path.join(args.src, "annotations")
    if not (os.path.isdir(src_img) and os.path.isdir(src_ann)):
        print(f"[错误] 目录结构不符合预期，需包含 images/ 与 annotations/\n  {args.src}")
        return 1

    # 输出目录
    out_img = os.path.join(args.out, "images", args.assign)
    out_lbl = os.path.join(args.out, "labels", args.assign)
    if not args.append:
        for d in (out_img, out_lbl):
            if os.path.isdir(d):
                shutil.rmtree(d)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    # 收集类别
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        found = []
        for f in os.listdir(src_ann):
            if f.lower().endswith(".xml"):
                _, _, objs = read_xml(os.path.join(src_ann, f))
                for o in objs:
                    if o[0] not in found:
                        found.append(o[0])
        classes = sorted(found)
    if not classes:
        print("[错误] 未从 XML 中解析到任何类别")
        return 1
    cid = {c: i for i, c in enumerate(classes)}
    print(f"类别: {classes}  →  {cid}")

    n_img = n_lbl = n_obj = n_skip = 0
    empty = []
    for f in sorted(os.listdir(src_img)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        xml_path = os.path.join(src_ann, stem + ".xml")
        if not os.path.exists(xml_path):
            print(f"  [跳过] 无标注: {f}")
            n_skip += 1
            continue

        w, h, objs = read_xml(xml_path)
        lines = []
        for name, x1, y1, x2, y2 in objs:
            # 裁剪到图像范围内
            x1 = max(0.0, min(x1, w))
            x2 = max(0.0, min(x2, w))
            y1 = max(0.0, min(y1, h))
            y2 = max(0.0, min(y2, h))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            lines.append(f"{cid[name]} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")

        if not lines:
            empty.append(f)
            n_skip += 1
            continue

        shutil.copy2(os.path.join(src_img, f), os.path.join(out_img, f))
        with open(os.path.join(out_lbl, stem + ".txt"), "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines) + "\n")
        n_img += 1
        n_lbl += len(lines)

    # classes.txt
    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(classes) + "\n")

    # dataset.yaml
    yaml_path = os.path.join(args.out, "dataset.yaml")
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    with open(yaml_path, "w", encoding="utf-8") as fp:
        fp.write(f"path: {os.path.abspath(args.out)}\n")
        fp.write("train: images/train\n")
        fp.write("val: images/val\n")
        fp.write("test: images/test\n")
        fp.write(f"names:\n{names}\n")

    print("-" * 56)
    print(f"转换完成：{n_img} 张 → images/{args.assign}，标注框 {n_lbl} 个")
    print(f"  跳过 {n_skip} 个（无标注或空框）")
    if empty:
        print(f"  空标注文件: {empty[:5]}{' ...' if len(empty) > 5 else ''}")
    print(f"  类别文件: {os.path.join(args.out, 'classes.txt')}")
    print(f"  数据集配置: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
