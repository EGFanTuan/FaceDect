[中文](README.zh-CN.md) | [English](README.md)

# Face Emotion Detection (YOLOv8)

A lightweight face emotion detection project built on Ultralytics YOLOv8. This repository includes training and inference scripts, along with a simple environment check to help you get started fast.

> Note on data and weights: To keep the repo small, datasets and intermediate model checkpoints are not tracked by git (ignored via .gitignore). We publish a full bundle (including dataset and all trained models) in the Releases section. If you download the Release bundle, you can run inference out of the box. If you cloned the source-only repository, you will need to provide dataset/weights yourself as described below.

## Requirements

- Python 3.8+
- Windows/Linux/macOS (GPU optional)
- Install dependencies:

```powershell
pip install -r .\requirements.txt
```

- (Optional) Verify your environment:

```powershell
python .\test_environment.py
```

## Quick Start

### 1) Inference (three modes)

Weights lookup order by default:
- `./emotion_detection/train_v1/weights/best.pt`
- `./best.pt`

You can also specify a custom path with `--model path/to/weights.pt`.

1) Images directory
- Put test images into `./input` (output will go to `./output`)
```powershell
python .\detect.py
```

2) Video file
```powershell
python .\detect.py --video .\path\to\video.mp4 --show --save-out .\output\detected_video.mp4
```

3) Webcam
```powershell
python .\detect.py --camera --cam-id 0 --show --save-out .\output\camera_out.mp4
```

For more runtime options (confidence, device, image size, etc.), see `DETECT_USAGE.md`.

### 2) Training

If you need to train:
- Download the dataset (see `datasets/emotion_dataset.yaml` for structure) and place it under `./datasets`.
- Optionally, download intermediate model checkpoints from Release if you want to resume training.

Run training with presets:
```powershell
# Default preset
python .\train.py

# Quick preset (fewer epochs, smaller batch, may use subset)
python .\train.py --config-preset quick

# High-quality preset (longer training)
python .\train.py --config-preset high-quality
```

Training artifacts are saved under `runs/detect/<experiment>/weights/` (e.g., `best.pt`, `last.pt`). For detailed configuration (batch size, epochs, device, optimizer, etc.), see `CONFIG_USAGE.md`.

## More Configuration

- Detector runtime options: `DETECT_USAGE.md`
- Training configuration and presets: `CONFIG_USAGE.md`

## Utilities

### Cache cleaner (datasets *.npy)

Clean .npy cache files under `./datasets/train` and `./datasets/val`.

PowerShell examples:

```powershell
# Preview (no deletion)
python .\ccache.py

# Actually delete (skip confirmation)
python .\ccache.py --no-dry-run -y

# Delete files older than 7 days
python .\ccache.py --older-than 7 --no-dry-run

# Include/Exclude patterns
python .\ccache.py --include *.cache.npy *.idx.npy
python .\ccache.py --exclude *keep*.npy --no-dry-run

# Verbose output
python .\ccache.py -v
```

## Releases and Assets

- Full bundles with dataset and all trained models are available in the Releases section.
- If you download a Release bundle, you can start inference immediately without extra setup.
- If you clone this repository only, you must:
  - Provide trained weights at one of the expected locations or via `--model`.
  - Provide the dataset under `./datasets` if you plan to train.

## Troubleshooting

- Run the environment check:
```powershell
python .\test_environment.py
```
- If no weights are found, specify them explicitly:
```powershell
python .\detect.py --model .\emotion_detection\train_v1\weights\best.pt
```

> Need more help? See the detailed guides: `CONFIG_USAGE.md` and `DETECT_USAGE.md`.

### Common issues table

| Symptom | Likely cause | How to fix |
| --- | --- | --- |
| ImportError: cv2/torch/ultralytics | Missing packages | pip install -r .\requirements.txt |
| No weights found error | Trained weights not present | Download the Release bundle or pass --model path\to\weights.pt |
| Dataset YAML/path errors during training | Dataset not placed under ./datasets or wrong structure | Follow datasets/emotion_dataset.yaml; ensure images/labels split exists |
| CUDA out of memory | Batch/img size too large for GPU memory | Reduce --batch or --imgsz; try --device cpu; consider half precision |
| OpenCV cannot open camera | Wrong camera id or permission | Try --cam-id 0/1; close other apps; check drivers |
| Video writer failed | Codec/permission/path issues | Ensure output folder exists; try different output path/filename |
| PyTorch/CUDA mismatch | Torch installed without matching CUDA | Reinstall torch for your CUDA, or use CPU device |

## FAQ

### What are the default training settings and expected performance?

- Dataset: current packaged dataset is a quarter of the original, with about ~27,000 training items.
- Reference machine: Ryzen 7 7735H + RTX 4060 Laptop + 32 GB RAM.
- Caching: default uses disk caching; using in-RAM caching (or disabling cache) may cause OOM on 32 GB RAM. If you hit OOM, consider shrinking the dataset further or using the quick preset.
- Runtime: 40 epochs take about ~5 hours on the reference machine.

Tips to adapt to your machine:
- Use presets to start: --config-preset quick for limited resources, high-quality for longer training.
- Reduce batch size and/or image size if you see OOM.
- Consider using dataset subsets; see CONFIG_USAGE.md for options and details.
- GPU users: ensure proper NVIDIA drivers and a matching PyTorch CUDA build.
