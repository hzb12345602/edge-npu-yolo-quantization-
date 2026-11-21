# -*- coding: utf-8 -*-
"""
环境自检脚本（论文1 训练/推理环境）
==================================
用途：检查当前 Python 环境是否满足
  ① 本地训练 YOLO11-n（ultralytics + torch）
  ② 黄金参考推理与误差分析（onnxruntime + opencv + numpy）

用法：
  D:\\anaconda3\\python.exe 论文\\code\\env_check.py          # 检查 base 环境
  conda run -n maixcam python 论文\\code\\env_check.py        # 检查指定环境

说明：脚本只读检查，不安装任何包；缺失项会在末尾给出安装命令。
"""
import importlib
import platform
import sys

print("=" * 62)
print(f"Python  : {sys.version.split()[0]}   ({sys.executable})")
print(f"平台    : {platform.system()} {platform.release()} / {platform.machine()}")
print("=" * 62)

GROUPS = {
    "训练依赖（本地训练 YOLO11-n）": [
        ("torch", "pip install torch --index-url https://download.pytorch.org/whl/cpu"),
        ("ultralytics", "pip install ultralytics==8.4.104"),
        ("yaml", "pip install pyyaml"),
        ("PIL", "pip install pillow"),
    ],
    "推理/分析依赖（黄金参考、误差分析）": [
        ("onnxruntime", "pip install onnxruntime"),
        ("onnx", "pip install onnx"),
        ("cv2", "pip install opencv-python"),
        ("numpy", "pip install numpy"),
        ("psutil", "pip install psutil"),
    ],
}

missing = []
for group, mods in GROUPS.items():
    print(f"\n【{group}】")
    for name, install_cmd in mods:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "")
            print(f"  [OK] {name:12s} {ver}")
        except Exception:
            print(f"  [缺失] {name:12s} → {install_cmd}")
            missing.append(install_cmd)

# torch 与 GPU 情况（决定本机训练速度）
print("\n【训练算力】")
try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"  torch: {torch.__version__}")
    print(f"  CUDA 可用: {cuda}" + ("" if cuda else "  → 将使用 CPU 训练（imgsz 320 + 100 张图可接受，建议夜间跑）"))
    if cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception:
    print("  torch 未安装，无法判断算力")

print("\n【ultralytics 版本要求】")
try:
    import ultralytics
    v = ultralytics.__version__
    ok = v == "8.4.104"
    print(f"  当前 {v} —— {'✅ 符合官方要求' if ok else '⚠️ 官方网页转换要求 8.4.104，建议对齐'}")
except Exception:
    print("  未安装 ultralytics")

print("\n" + "=" * 62)
if missing:
    print("缺失依赖，安装命令汇总：")
    for c in dict.fromkeys(missing):
        print("  " + c)
else:
    print("✅ 全部依赖齐备，可以开始训练与推理")
print("=" * 62)
