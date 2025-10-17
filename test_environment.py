#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick environment verification for the face emotion detection project.

This merged script performs:
- Dependency check (cv2, numpy, torch, ultralytics)
- Environment/version report (PyTorch/CUDA/GPU, OpenCV, Ultralytics)
- Directory check (./input, ./output)
- Model files check (common local paths)
- Model loading test via detect.EmotionDetector
"""

import os
import sys
from pathlib import Path
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Check required Python packages are installed."""
    logger.info("Checking required packages...")

    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'ultralytics': 'ultralytics',
    }

    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            logger.error(f"✗ {package} is NOT installed")
            missing_packages.append(pip_name)

    if missing_packages:
        logger.error("Missing packages detected. Install them with:")
        for pkg in missing_packages:
            logger.error(f"  pip install {pkg}")
        return False

    return True


def report_versions() -> bool:
    """Report versions of key libraries and GPU info (if available)."""
    logger.info("Reporting environment versions...")

    # PyTorch and CUDA
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        if cuda_available:
            try:
                logger.info(f"CUDA version: {getattr(torch.version, 'cuda', 'unknown')}")
            except Exception:
                logger.info("CUDA version: unknown")
            try:
                device_name = torch.cuda.get_device_name(0)
            except Exception:
                device_name = 'unknown'
            logger.info(f"GPU device: {device_name}")
    except Exception as e:
        logger.warning(f"Unable to report PyTorch/CUDA details: {e}")

    # OpenCV
    try:
        import cv2
        logger.info(f"OpenCV version: {cv2.__version__}")
    except Exception as e:
        logger.warning(f"Unable to report OpenCV version: {e}")

    # Ultralytics
    try:
        import ultralytics as ul
        version = getattr(ul, '__version__', 'unknown')
        logger.info(f"Ultralytics version: {version}")
    except Exception as e:
        logger.warning(f"Unable to report Ultralytics version: {e}")

    return True


def check_directories() -> bool:
    """Ensure required directories exist."""
    logger.info("Checking directory structure...")

    required_dirs = ['./input', './output']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"✓ Created directory: {dir_path}")
        else:
            logger.info(f"✓ Directory exists: {dir_path}")

    return True


def check_model_files() -> bool:
    """Check if local model files exist in common locations."""
    logger.info("Checking model files...")

    model_paths = [
        "./emotion_detection/train_v1/weights/best.pt",
        "./best.pt",
        "./emotion_detection/train_v1/weights/last.pt",
    ]

    found_models = []
    for model_path in model_paths:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"✓ Found model: {model_path} ({size_mb:.1f} MB)")
            found_models.append(model_path)
        else:
            logger.info(f"✗ Model not found: {model_path}")

    if not found_models:
        logger.error("No model files found. Please ensure training completed and weights are available.")
        return False

    # Recommend the first discovered model path for reference
    logger.info(f"Recommended model: {found_models[0]}")
    return True


def check_input_images() -> bool:
    """Check for test images in ./input."""
    logger.info("Checking input images...")

    input_dir = Path('./input')
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    if not input_dir.exists():
        logger.warning("./input directory does not exist")
        return False

    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        logger.warning("No image files found in ./input")
        logger.info("Please put some test images into ./input")
        return False

    logger.info(f"✓ Found {len(image_files)} images:")
    for img_file in image_files[:5]:  # show only first 5
        size_kb = img_file.stat().st_size / 1024
        logger.info(f"  - {img_file.name} ({size_kb:.1f} KB)")

    if len(image_files) > 5:
        logger.info(f"  ... and {len(image_files) - 5} more")

    return True


def test_model_loading() -> bool:
    """Try to load the emotion detection model via detect.EmotionDetector."""
    logger.info("Testing model loading...")

    try:
        from detect import EmotionDetector, get_model_path

        model_path = get_model_path()
        logger.info(f"Using model: {model_path}")

        detector = EmotionDetector(model_path, conf_threshold=0.5)
        logger.info("✓ Model loaded successfully!")

        # Report detector info
        try:
            labels = list(getattr(detector, 'emotion_labels', {}).values())
            if labels:
                logger.info(f"Supported emotion classes: {labels}")
        except Exception:
            pass
        logger.info(f"Confidence threshold: {getattr(detector, 'conf_threshold', 'n/a')}")
        logger.info(f"Inference device: {getattr(detector, 'device', 'n/a')}")

        return True

    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return False


def main() -> int:
    """Entrypoint for running all checks."""
    logger.info("=" * 60)
    logger.info("Face Emotion Detection - Environment Check")
    logger.info("=" * 60)

    checks = [
        ("Dependency Check", check_dependencies),
        ("Version Report", report_versions),
        ("Directory Check", check_directories),
        ("Model Files Check", check_model_files),
        ("Model Loading Test", test_model_loading),
    ]

    all_passed = True
    for check_name, check_func in checks:
        logger.info(f"\n{check_name}...")
        logger.info("-" * 30)
        try:
            result = check_func()
            if result:
                logger.info(f"✓ {check_name} passed")
            else:
                logger.warning(f"⚠ {check_name} has issues")
                if check_name in ["Dependency Check", "Model Files Check", "Model Loading Test"]:
                    all_passed = False
        except Exception as e:
            logger.error(f"✗ {check_name} failed: {e}")
            all_passed = False

    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✓ All checks passed! You can start using the detection script.")
        logger.info("\nTry:")
        logger.info("  python detect.py")
        logger.info("  python detect.py --help  # for detailed options")
        return 0
    else:
        logger.error("✗ Some checks failed. Please resolve the issues above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())