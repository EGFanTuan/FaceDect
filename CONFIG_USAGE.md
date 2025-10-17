# Configuration System Usage Examples

## Overview
The new configuration system replaces the complex argument parsing in `train.py` with a clean, structured approach using `config.py`.

## Basic Usage

### We highly recommend using configuration presets 'Default' for common scenarios. If you want to customize, use defaults and override specific parameters.

Here are some common usage examples:

### 1. Default Training(Most common)
```bash
python train.py
```

### 2. Quick Training (Fast preset)
```bash
python train.py --config-preset quick
```

### 3. High Quality Training(Needs a veeeeery strong device)
```bash
python train.py --config-preset high-quality
```

### 4. Custom Subset Training(Memory Efficient)
```bash
# Use 50% of dataset randomly
python train.py --use-subset --subset-ratio 0.5

# Use 30% of dataset with stratified sampling
python train.py --use-subset --subset-ratio 0.3 --subset-method stratified

# Use first 25% of dataset
python train.py --use-subset --subset-ratio 0.25 --subset-method first
```

### 5. Custom Parameters(Use defaults and override a few)
```bash
python train.py --epochs 100 --batch 32 --lr0 0.001 --experiment-name my_experiment
```

## Configuration Structure

### Training Configuration (`TrainingConfig`)
- **Model settings**: `model_name`, `num_classes`
- **Training params**: `epochs`, `batch_size`, `img_size`, `workers`
- **Optimizer**: `lr0`, `lrf`, `optimizer`, `weight_decay`
- **Strategy**: `patience`, `save_period`, `label_smoothing`
- **Hardware**: `device`
- **Output**: `project_name`, `experiment_name`

### Dataset Configuration (`DatasetConfig`)
- **Classes**: `emotion_classes` (angry, disgust, fear, happy, neutral, sad, surprise)
- **Splits**: `train_split`, `val_split`, `test_split`
- **Extensions**: `image_extensions`

### Inference Configuration (`InferenceConfig`)
- **Model**: `weights_path`, `conf_threshold`, `iou_threshold`
- **Input**: `source`, `img_size`
- **Output**: `save_results`, `save_dir`, `show_labels`

## Programmatic Usage

```python
from config import ConfigManager, get_quick_training_config

# Method 1: Use ConfigManager
config_manager = ConfigManager()
config_manager.update_training_config(epochs=80, batch_size=24)
train_config = config_manager.get_training_dict()

# Method 2: Use convenience functions
quick_config = get_quick_training_config(epochs=30, batch_size=16)

# Method 3: Direct configuration
from config import TrainingConfig
config = TrainingConfig()
config.epochs = 100
config.lr0 = 0.001
```

## Configuration Presets

### Default
- Epochs: 50
- Batch size: 16
- Learning rate: 0.01
- Patience: 10

### Quick
- Epochs: 20
- Batch size: 8
- Learning rate: 0.005
- Using subset of dataset (50%)
- For fast experimentation

### High-Quality
- Epochs: 100
- Batch size: 32
- Learning rate: 0.001
- Patience: 20
- For production models

## Dataset Sampling (Memory Management)

When dealing with large datasets that exceed memory limits, you can use subset training:

### Sampling Methods
- **`random`**: Randomly select files from the dataset
- **`first`**: Use the first N files (useful for consistent results)
- **`stratified`**: Maintain class distribution (recommended for balanced training)

### Memory Management Examples
```bash
# For very large datasets - use only 25% with balanced sampling
python train.py --use-subset --subset-ratio 0.25 --subset-method stratified

# Quick experimentation - use 10% of data
python train.py --config-preset quick --subset-ratio 0.1

# Memory-efficient preset (automatically configured)
python train.py --config-preset memory-efficient
```

### How It Works
1. Creates a new subset directory: `datasets_subset_0.5`
2. Copies sampled images and labels to subset directory
3. Generates new YAML configuration for subset
4. Trains on the subset dataset

## Dataset Download via YAML

Datasets are not bundled with the repository. To obtain data:

1. Open your dataset YAML under `./datasets/` (e.g., `datasets/emotion_dataset.yaml`).
2. Follow the `download` field (URL or script) to fetch the dataset.
3. Place it under `./datasets` following the YAML structure (`train/val/test` with `images/labels`).
4. If the dataset is still too large, use the subset options above to create a smaller training set.
