[中文](README.zh-CN.md) | [English](README.md)

# 人脸情绪检测 (YOLOv8)

基于 Ultralytics YOLOv8 的轻量级人脸情绪检测项目。仓库包含训练与推理脚本，并提供环境检测脚本，帮助你快速开始。

> 关于数据与权重：为避免仓库体积过大，数据集与训练过程中的中间模型均被 .gitignore 忽略，不随源码一起提交。Release 不再包含数据集。请根据 `./datasets/*.yaml` 中的 `download` 字段或相关链接自行下载数据集，并按 YAML 中的目录结构（`train/val/test` 与 `images/labels`）放置到 `./datasets` 下。部分 Release 可能提供已训练的模型权重，便于快速推理；如无则需自行提供权重。

## 环境要求

- Python 3.8+
- Windows/Linux/macOS（可选 GPU）
- 安装依赖：

```powershell
pip install -r .\requirements.txt
```

- （可选）环境检测：

```powershell
python .\test_environment.py
```

## 快速开始

### 1）推理（3 种方式）

默认权重查找顺序：
- `./emotion_detection/train_v1/weights/best.pt`
- `./best.pt`

也可通过 `--model path/to/weights.pt` 指定自定义权重路径。

1）图片目录
- 将测试图片放到 `./input`，结果输出到 `./output`
```powershell
python .\detect.py
```

2）视频文件
```powershell
python .\detect.py --video .\path\to\video.mp4 --show --save-out .\output\detected_video.mp4
```

3）摄像头
```powershell
python .\detect.py --camera --cam-id 0 --show --save-out .\output\camera_out.mp4
```

更多运行参数（置信度、设备、输入尺寸等）请参考 `DETECT_USAGE.md`。

### 2）训练

如果需要训练：
- 请自行下载数据集。打开 `datasets/emotion_dataset.yaml`（或你的数据集 YAML），根据其中的 `download` 字段或链接下载数据，并按 YAML 结构放置到 `./datasets` 目录。
- 如需从中断处继续或使用已有模型，可从 Release 下载训练权重（若有提供）。

使用预设启动训练：
```powershell
# 默认预设
python .\train.py

# 快速预设（更少 epoch、更小 batch，可能使用子集）
python .\train.py --config-preset quick

# 高质量预设（更长训练）
python .\train.py --config-preset high-quality
```

训练产物输出在 `runs/detect/<experiment>/weights/`（如 `best.pt`、`last.pt`）。若需详细配置（batch、epochs、device、optimizer 等），请查看 `CONFIG_USAGE.md`。

数据过大？可以使用子集训练以适配你的资源：
```powershell
# 使用前 25% 的数据
python .\train.py --use-subset --subset-ratio 0.25 --subset-method first

# 使用 30% 且保持类别分布
python .\train.py --use-subset --subset-ratio 0.3 --subset-method stratified

# quick 预设默认会使用子集
python .\train.py --config-preset quick
```

## 配置与文档

- 推理运行参数说明：`DETECT_USAGE.md`
- 训练配置与预设说明：`CONFIG_USAGE.md`

## 工具

### 缓存清理（datasets 下的 *.npy）

清理 `./datasets/train` 与 `./datasets/val` 下的 .npy 缓存文件。

PowerShell 示例：

```powershell
# 预览（不删除）
python .\ccache.py

# 实际删除（跳过确认）
python .\ccache.py --no-dry-run -y

# 仅删除 7 天前的文件
python .\ccache.py --older-than 7 --no-dry-run

# 包含/排除模式
python .\ccache.py --include *.cache.npy *.idx.npy
python .\ccache.py --exclude *keep*.npy --no-dry-run

# 详细输出
python .\ccache.py -v
```

## Release 与资源

- Release 不包含数据集。请根据 `./datasets/*.yaml` 中的 `download` 说明下载数据集。
- 部分 Release 可能提供已训练权重；如有，可加速你的推理或用于继续训练。
- 若仅克隆源码仓库，你需要：
  - 在默认路径放置已训练好的权重，或通过 `--model` 指定权重文件；
  - 若要训练，则需按 YAML 说明下载并放置数据集到 `./datasets`。

## 常见问题

- 运行环境检测：
```powershell
python .\test_environment.py
```
- 未找到权重时，可显式指定：
```powershell
python .\detect.py --model .\emotion_detection\train_v1\weights\best.pt
```

> 更多帮助请参阅详细文档：`CONFIG_USAGE.md` 与 `DETECT_USAGE.md`。

### 常见错误排查表

| 现象 | 可能原因 | 处理建议 |
| --- | --- | --- |
| ImportError: 找不到 cv2/torch/ultralytics | 依赖未安装 | pip install -r .\requirements.txt |
| No weights found / 未找到权重 | 未提供训练好的权重 | 下载 Release 包，或通过 --model 指定权重路径 |
| 训练时报数据集 YAML/路径错误 | 数据集未放在 ./datasets 或结构不符合 | 按 datasets/emotion_dataset.yaml 组织好 images/labels 及 train/val/test |
| CUDA out of memory | batch/输入尺寸过大 | 减小 --batch 或 --imgsz；尝试 --device cpu；或使用半精度 |
| OpenCV 无法打开摄像头 | 摄像头 ID 错误或权限占用 | 尝试 --cam-id 0/1；关闭占用摄像头的程序；检查驱动 |
| 视频写入失败 | 编码器/权限/路径问题 | 确保输出目录存在；更换输出文件名或路径 |
| PyTorch/CUDA 不匹配 | 安装了不匹配的 CUDA 版本 | 安装与 CUDA 匹配的 torch，或改用 CPU |

## FAQ

### 训练默认配置与性能说明

- 数据集：~~当前提供的数据集为原始数据集的 1/4，约 27,000 个训练样本~~ Release 不再包含数据集，请根据 `datasets/emotion_dataset.yaml` 中的 `download` 字段或链接自行下载数据集，并按 YAML 结构放置到 `./datasets` 下。
- 参考硬件：Ryzen7 7735H + RTX 4060 Laptop + 32 GB RAM。
- 缓存策略：默认使用磁盘缓存；若改为内存缓存（或关闭缓存）可能导致 32 GB 内存不足。如遇 OOM，建议进一步缩小数据集或使用 quick 预设。
- 参考耗时：在以上环境下使用原数据集的25%，40 个 epoch 训练约需 5 小时。

按你的设备进行调整的建议：
- 先用预设试跑：资源有限用 --config-preset quick，追求更高精度用 high-quality。
- 如遇 OOM，优先减小 batch 或 imgsz。
- 可使用数据子集进行训练；具体参数与方法见 CONFIG_USAGE.md。
- 使用 GPU 时，确保显卡驱动与 PyTorch CUDA 版本匹配。
