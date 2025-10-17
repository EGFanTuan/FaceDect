# 人脸情绪检测使用指南

## 功能概述

这个检测脚本可以：
- 检测本地图片中的人脸
- 识别人脸的情绪表情
- 在图片上绘制检测框和情绪标签
- 显示每个检测结果的置信度评分

## 使用方法

### 1. 基本用法

```powershell
# 检测 ./input 目录中的所有图片，结果保存到 ./output
python .\detect.py

# 或者使用完整参数
python .\detect.py --input .\input --output .\output
```

### 2. 自定义参数

```powershell
# 指定输入和输出目录
python .\detect.py -i .\path\to\input -o .\path\to\output

# 指定模型文件
python .\detect.py --model .\emotion_detection\train_v1\weights\best.pt

# 调整置信度阈值 (0.0-1.0)
python .\detect.py --conf 0.3

# 指定推理设备
python .\detect.py --device cpu     # 使用CPU
python .\detect.py --device 0       # 使用第一块GPU
```

### 3. 完整示例

```powershell
python .\detect.py `
    --input .\test_images `
    --output .\detection_results `
    --model .\best.pt `
    --conf 0.4 `
    --device 0
```

## 实时摄像头检测

支持直接从本机摄像头进行实时检测。

```powershell
# 使用默认摄像头(ID=0)，显示窗口
python .\detect.py --camera --show

# 指定摄像头ID、半精度与帧间隔（每2帧推理一次），并保存输出视频
python .\detect.py --camera --cam-id 0 --show --half --frame-stride 2 --save-out .\output\camera_out.mp4

# 指定推理尺寸、置信度与GPU
python .\detect.py --camera --imgsz 640 --conf 0.35 --device 0 --show
```

参数说明：
- `--camera` 启用摄像头模式
- `--cam-id` 选择摄像头编号（默认 0）
- `--show` 显示可视化窗口（按 `q` 或 `ESC` 退出）
- `--save-out` 保存实时结果为视频文件（如 .mp4）
- `--frame-stride` 帧间隔：每隔 N 帧推理一次，N>1 可显著提速
- `--imgsz` 推理输入尺寸（默认 640）
- `--half` 半精度推理，仅 CUDA 有效

## 视频文件检测

从已有视频文件中进行检测，支持显示和保存。

```powershell
# 边播边检测并显示预览，不保存
python .\detect.py --video .\input\demo.mp4 --show

# 保存输出视频到指定路径
python .\detect.py --video .\input\demo.mp4 --save-out .\output\demo_detected.mp4

# 未指定 --save-out 时，默认保存为 .\output\detected_<源文件名>.mp4
python .\detect.py --video .\input\demo.mp4
```

额外可选：
- `--frame-stride` 与摄像头相同，控制推理的帧间隔
- `--imgsz`、`--conf`、`--device`、`--half` 同摄像头参数一致

## 支持的图片格式

- JPG/JPEG
- PNG 
- BMP
- TIFF
- WebP

## 情绪类别

脚本可以检测以下7种情绪：
1. **angry** (愤怒) - 红色框
2. **disgust** (厌恶) - 深绿色框  
3. **fear** (恐惧) - 紫色框
4. **happy** (快乐) - 黄色框
5. **neutral** (中性) - 灰色框
6. **sad** (悲伤) - 蓝色框
7. **surprise** (惊讶) - 橙色框

## 输出说明

- 检测结果会保存为 `detected_原文件名.jpg` 格式
- 每个检测框会显示：情绪标签和置信度评分
- 控制台会输出详细的检测信息

## 注意事项

1. **预处理与缩放**：已由 YOLO 内置 letterbox 自动处理，无需手动缩放
2. **模型路径**：脚本会自动查找最佳模型文件
3. **设备选择**：建议使用GPU以获得更快的检测速度
4. **置信度阈值**：可以根据需要调整以过滤低置信度的检测结果
5. **半精度**：`--half` 仅在 CUDA 可用时生效；CPU 情况会自动忽略
6. **退出方式**：实时/视频预览窗口按 `q` 或 `ESC` 退出
