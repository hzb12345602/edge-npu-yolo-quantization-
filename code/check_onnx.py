# -*- coding: utf-8 -*-
"""
ONNX 模型自检（黄金参考前置检查）
================================
检查项：
  1. 输入名与输入形状（应为 [1, 3, 224, 320]，即 NCHW 高 224 × 宽 320）
  2. 输出张量数量与形状（应为检测头原始输出，无 NMS）
  3. 用随机张量跑一次推理，确认可运行

用法：
  python check_onnx.py <模型路径.onnx>
"""
import sys

import numpy as np
import onnxruntime as ort

sys.stdout.reconfigure(encoding="utf-8")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "best.onnx"
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print(f"模型: {path}")
    print("-" * 58)
    ins = sess.get_inputs()
    for i in ins:
        print(f"输入 : {i.name:20s} shape={i.shape}  type={i.type}")
    outs = sess.get_outputs()
    for o in outs:
        print(f"输出 : {o.name:20s} shape={o.shape}")
    print("-" * 58)

    shape = [d if isinstance(d, int) else 1 for d in ins[0].shape]
    if len(shape) != 4:
        print("⚠️ 输入不是 4 维 NCHW，请检查导出参数")
        return
    n, c, h, w = shape
    if (h, w) == (224, 320):
        print(f"✅ 输入尺寸符合 MaixCAM/Pro 部署要求：高 {h} × 宽 {w}")
    else:
        print(f"⚠️ 输入尺寸为 高 {h} × 宽 {w}；MaixCAM/Pro 官方推荐导出 imgsz=224,320")
    if len(outs) == 1 and "output0" in outs[0].name:
        print("✅ 输出为单检测头原始张量（无 NMS 后处理），与板端 mud 后处理一致")
    else:
        print(f"ℹ️ 输出张量数：{len(outs)}（请确认无 NMS 后处理）")

    x = np.random.rand(n, c, h, w).astype(np.float32)
    y = sess.run(None, {ins[0].name: x})
    print(f"✅ 推理成功，输出张量形状: {[t.shape for t in y]}")


if __name__ == "__main__":
    main()
