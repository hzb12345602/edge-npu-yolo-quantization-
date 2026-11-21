# 实验流程与方法

> 本文档描述本项目的完整实验管线，与 `code/` 下的脚本一一对应，供方法复现参考。
> 命令中的绝对路径已替换为 `<占位>`，请按你的实际环境替换。
> 论文全文、数据集与模型权重见私有仓库；本仓库只提供方法与代码。

---

## 0. 总览

```
数据采集与标注 → 困难样本三维度筛选 → 校准集构建(困难60 vs 随机60)
  → 本地训练 YOLO11-n → 导出 ONNX(224×320) → TPU-MLIR 量化编译(2 个 cvimodel)
  → 板端部署实测(延迟/FPS/资源) → FP32 黄金参考对比 → 量化误差/延迟分解 → 结果可视化
```

共 8 个阶段，端到端在实体边缘硬件 **MaixCAM-Pro** 上完成。

---

## 1. 数据集

- 规模：**269 张**，单类（类名"手柄"），YOLO 标注格式，每张 1 个目标框。
- 划分：训练 181 / 验证 43 / 测试 45，**按拍摄段划分而非随机划分**，避免连续相似帧跨集造成数据泄漏。
- 分辨率：拍摄 320×320；部署输入 320×224（长宽比不一致 → 每帧做 letterbox，见 §8）。
- 转换脚本：`code/voc2yolo.py`（VOC → YOLO，支持 train/val/test 与追加）。

## 2. 困难样本三维度筛选

用三个与检测鲁棒性强相关的维度刻画"困难程度"，并对训练集逐张统计：

| 维度 | 定义 | 阈值 |
| --- | --- | --- |
| 小目标 | 目标归一化面积 | **< 0.02** |
| 低光照 | 图像平均亮度 | **< 0.30** |
| 模糊 | Laplacian 方差 | 取数据集清晰度分布的 **20% 分位数**（本数据集实测 48，不取固定经验值 100） |

分桶按优先级「小目标 > 低光照 > 模糊 > 普通」，每个样本只归入最高优先级的桶。
脚本：`code/analyze_dataset.py`（统计 + 分桶）、`code/screen_hard_samples.py`（筛选导出）。

## 3. 校准集构建

- **实验组（困难混合校准）**：小目标 / 低光照 / 模糊各 20 张，共 **60 张**。
- **对照组（随机校准）**：从训练集随机采样 **60 张**。
- 两组规模一致、仅分布不同，从而隔离"校准集构造策略"这一变量。
- 脚本：`code/random_calib.py`。

## 4. 训练与导出

```bash
# 训练：Ultralytics 8.4.104，YOLO11-n，150 轮，训练输入 320×320
yolo detect train model=yolo11n.pt data=<dataset.yaml> epochs=150 imgsz=320

# 导出：部署输入 224×320（高×宽），opset 17，关闭动态，开启 simplify
yolo export model=<best.pt> format=onnx imgsz=224,320 opset=17 simplify=True dynamic=False
```

导出后**裁剪输出节点**，只保留检测头解码所需的 `dfl/Conv` 与 `Sigmoid` 两个输出（板端后处理按此约定执行）。脚本：`code/check_onnx.py`（校验输入 `[1,3,224,320]`、输出形状）。

## 5. 量化编译（TPU-MLIR）

在 `sophgo/tpuc_dev` 容器内执行三件套，两组校准各产出 1 个 cvimodel：

```bash
# ① ONNX → MLIR：预处理参数必须与部署/黄金参考逐项一致
model_transform.py --model_name yolo11n --model_def <best.onnx> \
  --input_shapes [[1,3,224,320]] --mean 0,0,0 --scale 0.00392156862745098 \
  --pixel_format rgb --pad_value 114 --output_names <裁剪后输出> --mlir yolo11n.mlir

# ② 校准：困难混合 60 张 vs 随机 60 张
run_calibration.py yolo11n.mlir --dataset <hard_calib>   --input_num 60 -o hard_cali_table
run_calibration.py yolo11n.mlir --dataset <random_calib> --input_num 60 -o random_cali_table

# ③ 部署：INT8，cv181x
model_deploy.py --mlir yolo11n.mlir --quantize INT8 --calibration_table hard_cali_table   --processor cv181x --model yolo11n_hard_int8.cvimodel
model_deploy.py --mlir yolo11n.mlir --quantize INT8 --calibration_table random_cali_table --processor cv181x --model yolo11n_random_int8.cvimodel
```

## 6. 黄金参考与量化误差分析

- **黄金参考**：Windows 端 ONNXRuntime 加载 FP32 ONNX，`imgsz=(224,320)`、`conf=0.25 / iou=0.45`，与板端预处理（RGB、letterbox、pad=114、×1/255）逐项一致。
- **对比**：板端 INT8 输出与黄金参考按图像做 **IoU ≥ 0.5 的贪心一对一匹配**。
- **指标**：坐标 MAE / MaxAE / RMSE、超阈值（≥2 px）占比、中心点误差、归一化误差（÷目标对角线）、置信度偏差、匹配 IoU。
- **分层**：按 §2 阈值把测试集分层为 小目标 / 低光照 / 模糊 / 普通，逐层统计。
- 脚本：`code/board_analysis.py`（golden / compare / map / metrics 四步）。

## 7. 板端实测

- 平台：MaixCAM-Pro（SG2002，1 TOPS INT8 NPU，实测频点 700 MHz），MaixPy v4.10.3。
- 部署格式：`.mud`（INI）+ `.cvimodel`；后处理 API `nn.YOLO11`。
- 采集：逐张推理 45 张测试图，`dual_buff=False`（真实单帧延迟口径），剔除 3 帧预热后统计 42 帧；记录延迟 / FPS / CPU / 内存 / 温度 + 检测框。
- 脚本：`code/board_collect.py`。

## 8. 端到端延迟分解

固件把 YOLO 前后处理实现为 C++ 内置类，Python 层不暴露子步骤，故用两类可实测口径分解：

1. **输入尺寸 A/B 对照**：同一模型分别推理原始 320×320 与预缩放 320×224 图像，差值 = letterbox 预处理代价；
2. **通用推理接口对照**：`maix.nn.NN.forward_image`（只做预处理 + 推理，不含 YOLO 解码与 NMS）与 `detect()` 的差值 = 后处理代价。

脚本：`code/board_latency_breakdown.py`、`code/board_probe_api.py`。

## 9. 结果可视化

`code/make_figures.py` 从原始结果 CSV 直接出图（不手工描点），产物对应本仓库 `figures/` 下 7 张图。关键数字见 [README.md](README.md)。
