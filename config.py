# config.py
"""
YOLOv8 Emotion Detection Training Configuration
This file contains all configuration parameters for training, validation, and inference.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class TrainingConfig:
    """Training configuration class"""
    
    # Dataset configuration
    data_path: str = './datasets/emotion_dataset.yaml'
    dataset_root: str = './datasets'
    
    # Model configuration
    model_name: str = 'yolov8s.pt'
    num_classes: int = 7
    
    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    workers: int = 4
    
    # Optimizer settings
    lr0: float = 0.01  # Initial learning rate
    lrf: float = 0.01  # Final learning rate (lr0 * lrf)
    optimizer: str = 'auto'  # SGD, Adam, AdamW, etc.
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    cos_lr: bool = False  # Use cosine learning rate scheduler
    
    # Training strategy
    patience: int = 10  # Early stopping patience
    save_period: int = 10  # Save checkpoint every n epochs
    label_smoothing: float = 0.0
    cache: str = 'disk'  # Cache images in memory (ram, disk, or False)
    
    # Dataset sampling options
    use_subset: bool = False  # Whether to use only a subset of the dataset
    subset_ratio: float = 0.5  # Fraction of dataset to use (0.1-1.0)
    subset_method: str = 'random'  # 'random', 'first', or 'stratified'
    
    # Hardware settings
    device: str = '0'  # Device to use for training (e.g., 0,1 for GPU or "cpu")
    
    # Output settings
    project_name: str = 'emotion_detection'
    experiment_name: str = 'train_v1'
    exist_ok: bool = True
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for YOLO training"""
        return {
            'data': self.data_path,
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.img_size,
            'workers': self.workers,
            'device': self.device,
            'lr0': self.lr0,
            'lrf': self.lrf,
            'patience': self.patience,
            'save_period': self.save_period,
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'cos_lr': self.cos_lr,
            'label_smoothing': self.label_smoothing,
            'cache': self.cache,
            'verbose': self.verbose,
            'project': self.project_name,
            'name': self.experiment_name,
            'exist_ok': self.exist_ok,
        }
    
    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        if hasattr(args, 'data') and args.data:
            self.data_path = args.data
        if hasattr(args, 'model') and args.model:
            self.model_name = args.model
        if hasattr(args, 'epochs') and args.epochs:
            self.epochs = args.epochs
        if hasattr(args, 'batch') and args.batch:
            self.batch_size = args.batch
        if hasattr(args, 'img_size') and args.img_size:
            self.img_size = args.img_size
        if hasattr(args, 'workers') and args.workers:
            self.workers = args.workers
        if hasattr(args, 'lr0') and args.lr0:
            self.lr0 = args.lr0
        if hasattr(args, 'lrf') and args.lrf:
            self.lrf = args.lrf
        if hasattr(args, 'patience') and args.patience:
            self.patience = args.patience
        if hasattr(args, 'save_period') and args.save_period:
            self.save_period = args.save_period
        if hasattr(args, 'device') and args.device:
            self.device = args.device
        if hasattr(args, 'optimizer') and args.optimizer:
            self.optimizer = args.optimizer
        if hasattr(args, 'weight_decay') and args.weight_decay:
            self.weight_decay = args.weight_decay
        if hasattr(args, 'warmup_epochs') and args.warmup_epochs:
            self.warmup_epochs = args.warmup_epochs
        if hasattr(args, 'cos_lr') and args.cos_lr:
            self.cos_lr = args.cos_lr
        if hasattr(args, 'label_smoothing') and args.label_smoothing:
            self.label_smoothing = args.label_smoothing
        if hasattr(args, 'cache') and args.cache:
            self.cache = args.cache
        if hasattr(args, 'use_subset') and args.use_subset:
            self.use_subset = args.use_subset
        if hasattr(args, 'subset_ratio') and args.subset_ratio:
            self.subset_ratio = args.subset_ratio
        if hasattr(args, 'subset_method') and args.subset_method:
            self.subset_method = args.subset_method


@dataclass
class DatasetConfig:
    """Dataset configuration class"""
    
    # Emotion classes (modify according to your dataset)
    emotion_classes: List[str] = None
    
    # Dataset splits
    train_split: str = 'train'
    val_split: str = 'val'
    test_split: str = 'test'
    
    # Image extensions
    image_extensions: List[str] = None
    
    def __post_init__(self):
        if self.emotion_classes is None:
            self.emotion_classes = [
                'angry',
                'disgust', 
                'fear',
                'happy',
                'neutral',
                'sad',
                'surprise'
            ]
        
        if self.image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    @property
    def num_classes(self) -> int:
        return len(self.emotion_classes)
    
    def get_yaml_content(self, dataset_path: str = './datasets') -> str:
        """Generate YAML content for YOLO dataset configuration"""
        yaml_content = f"""# Emotion Detection Dataset
path: {dataset_path}  # dataset root dir
train: {self.train_split}/images  # train images (relative to 'path')
val: {self.val_split}/images    # val images (relative to 'path')
test: {self.test_split}/images  # test images (relative to 'path')

# Number of classes
nc: {self.num_classes}  # number of classes

# Class names
names: 
"""
        # Add class names with indices
        for i, class_name in enumerate(self.emotion_classes):
            yaml_content += f"  {i}: {class_name}\n"
        
        yaml_content += """
# Download script/URL (optional)
# download: https://ultralytics.com/assets/coco128.zip
"""
        return yaml_content


@dataclass
class InferenceConfig:
    """Inference configuration class"""
    
    # Model settings
    weights_path: str = 'runs/detect/train_v1/weights/best.pt'
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    
    # Input settings
    source: str = '0'  # Camera index, image path, or video path
    img_size: int = 640
    
    # Output settings
    save_results: bool = True
    save_dir: str = 'runs/detect/inference'
    show_labels: bool = True
    show_conf: bool = True
    
    # Device settings
    device: str = '0'  # GPU device or 'cpu'


class ConfigManager:
    """Configuration manager to handle different config scenarios"""
    
    def __init__(self):
        self.training_config = TrainingConfig()
        self.dataset_config = DatasetConfig()
        self.inference_config = InferenceConfig()
    
    def update_training_config(self, **kwargs):
        """Update training configuration with provided arguments"""
        for key, value in kwargs.items():
            if hasattr(self.training_config, key):
                setattr(self.training_config, key, value)
    
    def update_inference_config(self, **kwargs):
        """Update inference configuration with provided arguments"""
        for key, value in kwargs.items():
            if hasattr(self.inference_config, key):
                setattr(self.inference_config, key, value)
    
    def get_training_dict(self) -> Dict[str, Any]:
        """Get training configuration as dictionary"""
        return self.training_config.to_dict()
    
    def create_dataset_yaml(self, output_path: Optional[str] = None) -> str:
        """Create dataset YAML file"""
        if output_path is None:
            output_path = self.training_config.data_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write YAML content
        yaml_content = self.dataset_config.get_yaml_content(self.training_config.dataset_root)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        return output_path
    
    def create_subset_dataset(self, original_dataset_path: str = None) -> str:
        """Create a subset of the original dataset for training"""
        import random
        import shutil
        from collections import defaultdict
        
        if original_dataset_path is None:
            original_dataset_path = self.training_config.dataset_root
        
        subset_path = f"{original_dataset_path}_subset_{self.training_config.subset_ratio}"
        
        print(f"📁 Creating subset dataset at: {subset_path}")
        print(f"   Subset ratio: {self.training_config.subset_ratio}")
        print(f"   Sampling method: {self.training_config.subset_method}")
        
        # Create subset directory structure
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(subset_path, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(subset_path, split, 'labels'), exist_ok=True)
        
        # Sample data for each split
        for split in ['train', 'val', 'test']:
            original_images_dir = os.path.join(original_dataset_path, split, 'images')
            original_labels_dir = os.path.join(original_dataset_path, split, 'labels')
            subset_images_dir = os.path.join(subset_path, split, 'images')
            subset_labels_dir = os.path.join(subset_path, split, 'labels')
            
            if not os.path.exists(original_images_dir):
                print(f"   Skipping {split} - directory not found")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(original_images_dir) 
                          if f.lower().endswith(tuple(self.dataset_config.image_extensions))]
            
            if not image_files:
                print(f"   Skipping {split} - no images found")
                continue
            
            # Sample files based on method
            num_samples = int(len(image_files) * self.training_config.subset_ratio)
            num_samples = max(1, num_samples)  # At least 1 sample
            
            if self.training_config.subset_method == 'random':
                sampled_files = random.sample(image_files, num_samples)
            elif self.training_config.subset_method == 'first':
                sampled_files = image_files[:num_samples]
            elif self.training_config.subset_method == 'stratified':
                # Group by emotion class (assuming filename contains emotion)
                emotion_groups = defaultdict(list)
                for img_file in image_files:
                    # Extract emotion from filename (assuming format like "emotion_...")
                    for emotion in self.dataset_config.emotion_classes:
                        if emotion.lower() in img_file.lower():
                            emotion_groups[emotion].append(img_file)
                            break
                    else:
                        emotion_groups['unknown'].append(img_file)
                
                # Sample proportionally from each group
                sampled_files = []
                for emotion, files in emotion_groups.items():
                    if files:
                        group_samples = max(1, int(len(files) * self.training_config.subset_ratio))
                        sampled_files.extend(random.sample(files, min(group_samples, len(files))))
                
                # If we have too many samples, randomly reduce
                if len(sampled_files) > num_samples:
                    sampled_files = random.sample(sampled_files, num_samples)
            else:
                sampled_files = random.sample(image_files, num_samples)
            
            # Copy sampled files
            copied_count = 0
            for img_file in sampled_files:
                # Copy image
                src_img = os.path.join(original_images_dir, img_file)
                dst_img = os.path.join(subset_images_dir, img_file)
                shutil.copy2(src_img, dst_img)
                
                # Copy corresponding label if exists
                label_file = os.path.splitext(img_file)[0] + '.txt'
                src_label = os.path.join(original_labels_dir, label_file)
                if os.path.exists(src_label):
                    dst_label = os.path.join(subset_labels_dir, label_file)
                    shutil.copy2(src_label, dst_label)
                
                copied_count += 1
            
            print(f"   {split.upper():5}: {copied_count:4} / {len(image_files):4} files copied")
        
        # Update dataset path to use subset
        subset_yaml_path = os.path.join(subset_path, 'emotion_dataset.yaml')
        yaml_content = self.dataset_config.get_yaml_content(subset_path)
        with open(subset_yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        # Update training config to use subset
        self.training_config.data_path = subset_yaml_path
        self.training_config.dataset_root = subset_path
        
        print(f"✅ Subset dataset created successfully!")
        return subset_path
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("=== Configuration Summary ===")
        print(f"Model: {self.training_config.model_name}")
        print(f"Epochs: {self.training_config.epochs}")
        print(f"Batch Size: {self.training_config.batch_size}")
        print(f"Image Size: {self.training_config.img_size}")
        print(f"Learning Rate: {self.training_config.lr0}")
        print(f"Device: {self.training_config.device}")
        print(f"Classes: {self.dataset_config.num_classes}")
        print(f"Dataset: {self.training_config.data_path}")
        if self.training_config.use_subset:
            print(f"Subset Mode: {self.training_config.subset_ratio*100:.0f}% ({self.training_config.subset_method})")
        print("=" * 30)


# Create default configuration instance
default_config = ConfigManager()


# Convenience functions
def get_default_training_config() -> TrainingConfig:
    """Get default training configuration"""
    return TrainingConfig()

def get_default_dataset_config() -> DatasetConfig:
    """Get default dataset configuration"""
    return DatasetConfig()

def get_default_inference_config() -> InferenceConfig:
    """Get default inference configuration"""
    return InferenceConfig()

def get_quick_training_config(
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    device: str = '0'
) -> TrainingConfig:
    """Get quickly configured training config with common parameters"""
    config = TrainingConfig()
    config.epochs = epochs
    config.batch_size = batch_size
    config.lr0 = learning_rate
    config.device = device
    return config


if __name__ == "__main__":
    # Test configuration
    config_manager = ConfigManager()
    config_manager.print_config_summary()
    
    # Create sample dataset YAML
    yaml_path = config_manager.create_dataset_yaml('./datasets/emotion_dataset.yaml')
    print(f"Created dataset YAML: {yaml_path}")
