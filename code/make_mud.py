# -*- coding: utf-8 -*-
"""
生成 MaixPy 可识别的 mud 描述文件（INI 格式）
==========================================
⚠️ MaixPy 的 mud 是 **INI** 格式，不是 JSON（JSON 会报 "parse model ... failed, err 5"）

参考官方格式（Sipeed Wiki《手动转换给 MaixCAM 用》）：
    [basic]
    type = cvimodel
    model = xxx.cvimodel

    [extra]
    model_type = yolo11            # 支持的取值：yolov5 / yolov8 / yolo11 / classifier
    input_type = rgb               # 必须与转换、训练导出一致
    mean = 0, 0, 0
    scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
    labels = 手柄

用法：
  python make_mud.py --model yolo11n_hard_int8.cvimodel --out D:\\...\\yolo11n_hard.mud --labels 手柄
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

TEMPLATE = """[basic]
type = cvimodel
model = {model}

[extra]
model_type = {model_type}
input_type = {input_type}
mean = {mean}
scale = {scale}
labels = {labels}
"""


def parse_args():
    ap = argparse.ArgumentParser(description="生成 MaixPy mud 描述文件（INI 格式）")
    ap.add_argument("--model", required=True, help="cvimodel 文件名（与 mud 同目录，只写文件名）")
    ap.add_argument("--out", required=True, help="输出 mud 路径")
    ap.add_argument("--model-type", default="yolo11", help="yolov5 / yolov8 / yolo11 / classifier")
    ap.add_argument("--input-type", default="rgb", help="rgb 或 bgr（须与转换一致）")
    ap.add_argument("--labels", default="手柄", help="类别名，多个用英文逗号分隔")
    ap.add_argument("--mean", default="0, 0, 0")
    ap.add_argument("--scale",
                    default="0.00392156862745098, 0.00392156862745098, 0.00392156862745098")
    return ap.parse_args()


def main():
    a = parse_args()
    if not a.model.lower().endswith(".cvimodel"):
        print("[警告] model 建议填 .cvimodel 文件名")
    txt = TEMPLATE.format(model=a.model, model_type=a.model_type,
                          input_type=a.input_type, mean=a.mean,
                          scale=a.scale, labels=a.labels)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    print(f"[完成] 已生成: {a.out}\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
