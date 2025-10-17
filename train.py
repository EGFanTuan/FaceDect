# train.py
import os
import argparse
import logging
import datetime
import json
import psutil
import sys
from ultralytics import YOLO
import torch
from config import ConfigManager

def setup_logging():
    """
    Setup comprehensive logging for training monitoring
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup main logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/training_{timestamp}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Log system information
    logger.info("="*60)
    logger.info("TRAINING SESSION STARTED")
    logger.info("="*60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Log system resources
    memory = psutil.virtual_memory()
    logger.info(f"Total RAM: {memory.total / 1e9:.1f} GB")
    logger.info(f"Available RAM: {memory.available / 1e9:.1f} GB")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    
    return logger

def save_training_checkpoint(config_manager, epoch=0, status="starting"):
    """
    Save training checkpoint information for recovery
    """
    checkpoint_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "epoch": epoch,
        "status": status,
        "config": {
            "model_name": config_manager.training_config.model_name,
            "epochs": config_manager.training_config.epochs,
            "batch_size": config_manager.training_config.batch_size,
            "experiment_name": config_manager.training_config.experiment_name
        },
        "system_info": {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "total_memory_gb": psutil.virtual_memory().total / 1e9
        }
    }
    
    checkpoint_file = "training_checkpoint.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

def check_previous_training():
    """
    Check if there was a previous training session that was interrupted
    """
    checkpoint_file = "training_checkpoint.json"
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            if checkpoint.get("status") != "completed":
                print("🔍 检测到之前的训练可能被中断:")
                print(f"   时间: {checkpoint.get('timestamp', 'Unknown')}")
                print(f"   实验: {checkpoint.get('config', {}).get('experiment_name', 'Unknown')}")
                print(f"   状态: {checkpoint.get('status', 'Unknown')}")
                print(f"   轮数: {checkpoint.get('epoch', 0)}")
                
                # Check if there are saved weights
                exp_name = checkpoint.get('config', {}).get('experiment_name', '')
                if exp_name:
                    weights_dir = f"runs/detect/{exp_name}/weights"
                    if os.path.exists(weights_dir):
                        weights_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
                        if weights_files:
                            print(f"   找到权重文件: {weights_files}")
                            response = input("是否要从上次中断的地方继续训练? (y/n): ")
                            if response.lower() == 'y':
                                return weights_dir + "/last.pt"
                
        except Exception as e:
            print(f"读取checkpoint文件时出错: {e}")
    
    return None

def setup_training_config():
    """
    Setup training configuration and arguments (simplified version using config.py)
    """
    parser = argparse.ArgumentParser(description='YOLOv8 Emotion Detection Training')
    
    # Essential arguments that might need to be overridden
    parser.add_argument('--config-preset', type=str, default='default', 
                       choices=['default', 'quick', 'high-quality', 'memory-efficient'],
                       help='Configuration preset to use')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--model', type=str, help='Pretrained model to start from')
    parser.add_argument('--device', type=str, help='Device to use for training')
    parser.add_argument('--lr0', type=float, help='Initial learning rate')
    parser.add_argument('--experiment-name', type=str, help='Experiment name for outputs')
    
    # Dataset sampling arguments (for memory management)
    parser.add_argument('--use-subset', action='store_true', 
                       help='Use only a subset of the dataset for training')
    parser.add_argument('--subset-ratio', type=float, default=0.5, 
                       help='Fraction of dataset to use (0.1-1.0)')
    parser.add_argument('--subset-method', type=str, default='random',
                       choices=['random', 'first', 'stratified'],
                       help='Method to sample subset: random, first N, or stratified')
    
    # Advanced arguments (optional overrides)
    parser.add_argument('--data', type=str, help='Path to dataset YAML file')
    parser.add_argument('--img-size', type=int, help='Image size for training')
    parser.add_argument('--workers', type=int, help='Number of data loading workers')
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, help='Optimizer to use')
    parser.add_argument('--weight-decay', type=float, help='Optimizer weight decay')
    parser.add_argument('--cos-lr', action='store_true', help='Use cosine learning rate scheduler')
    parser.add_argument('--cache', type=str, help='Cache images in memory')
    
    return parser.parse_args()

def create_dataset_yaml(config_manager):
    """
    Create dataset YAML file using configuration manager
    """
    yaml_path = config_manager.training_config.data_path
    if not os.path.exists(yaml_path):
        yaml_path = config_manager.create_dataset_yaml()
        print(f"Created dataset YAML file: {yaml_path}")
    
    return yaml_path

def check_dataset_structure():
    """
    Verify dataset structure and count samples
    """
    dataset_path = './datasets'
    splits = ['train', 'val', 'test']
    
    print("=== Dataset Structure Check ===")
    for split in splits:
        images_dir = os.path.join(dataset_path, split, 'images')
        labels_dir = os.path.join(dataset_path, split, 'labels')
        
        if os.path.exists(images_dir):
            images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')] if os.path.exists(labels_dir) else []
            
            print(f"{split.upper():6}: {len(images):6} images, {len(labels):6} labels")
            
            # Check for label-file correspondence
            if len(images) != len(labels):
                print(f"  WARNING: Mismatch between images and labels in {split}")
        else:
            print(f"{split.upper():6}: Directory not found - {images_dir}")
    
    print("=" * 40)

def train_model(config_manager):
    """
    Main training function using configuration manager
    """
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting YOLOv8 Emotion Detection Training...")
    
    # Save initial checkpoint
    save_training_checkpoint(config_manager, 0, "starting")
    
    # Check original dataset structure
    check_dataset_structure()
    
    # Create subset dataset if requested
    if config_manager.training_config.use_subset:
        logger.info(f"📊 Creating subset dataset ({config_manager.training_config.subset_ratio*100:.0f}% of original)...")
        subset_path = config_manager.create_subset_dataset()
        logger.info(f"   Subset created at: {subset_path}")
        
        # Check subset dataset structure
        logger.info("=== Subset Dataset Structure ===")
        original_dataset_root = config_manager.training_config.dataset_root
        config_manager.training_config.dataset_root = subset_path
        check_dataset_structure()
        config_manager.training_config.dataset_root = original_dataset_root
        logger.info("=" * 40)
    
    # Create dataset YAML if needed
    yaml_path = create_dataset_yaml(config_manager)
    
    # Check if we should resume from previous training
    resume_from = check_previous_training()
    
    # Load model
    model_name = config_manager.training_config.model_name if not resume_from else resume_from
    logger.info(f"📦 Loading model: {model_name}")
    if resume_from:
        logger.info(f"🔄 Resuming training from: {resume_from}")
    
    model = YOLO(model_name)
    
    # Get training configuration as dictionary
    train_config = config_manager.get_training_dict()
    
    # Set resume flag if needed
    if resume_from:
        train_config['resume'] = True
    
    logger.info("⚙️ Training Configuration:")
    for key, value in train_config.items():
        logger.info(f"  {key}: {value}")
    
    # Start training
    logger.info("🎯 Starting training...")
    try:
        # Update checkpoint to running
        save_training_checkpoint(config_manager, 0, "running")
        
        results = model.train(**train_config)
        
        # Update checkpoint to completed
        save_training_checkpoint(config_manager, 
                               config_manager.training_config.epochs, 
                               "completed")
        
        logger.info("✅ Training completed successfully!")
        return results
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        # Update checkpoint to failed
        save_training_checkpoint(config_manager, 0, "failed")
        return None

def main():
    """
    Main execution function using configuration manager
    """
    # Parse command line arguments
    args = setup_training_config()
    
    # Initialize configuration manager with preset
    from config import ConfigManager, get_quick_training_config
    
    if args.config_preset == 'quick':
        # Quick training preset (fewer epochs, smaller batch, use subset)
        config_manager = ConfigManager()
        config_manager.training_config = get_quick_training_config(
            epochs=20, batch_size=8, learning_rate=0.005
        )
        # Automatically use subset for quick training to save memory
        config_manager.training_config.use_subset = True
        config_manager.training_config.subset_ratio = 0.5  # Use only 30% for quick training
    elif args.config_preset == 'high-quality':
        # High quality preset (more epochs, optimized parameters)
        config_manager = ConfigManager()
        config_manager.training_config.epochs = 100
        config_manager.training_config.batch_size = 32
        config_manager.training_config.patience = 20
        config_manager.training_config.lr0 = 0.001
    else:
        # Default preset
        print("Using default training configuration.")
        config_manager = ConfigManager()
    
    # Override configuration with command line arguments
    config_manager.training_config.update_from_args(args)
    
    # Update experiment name if provided
    if args.experiment_name:
        config_manager.training_config.experiment_name = args.experiment_name
    
    # Check CUDA availability and set device
    device = config_manager.training_config.device
    if device != 'cpu' and torch.cuda.is_available():
        print(f"🎯 CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU for training")
        config_manager.training_config.device = 'cpu'
    
    # Print configuration summary
    config_manager.print_config_summary()
    
    # Start training
    results = train_model(config_manager)
    
    if results:
        print("\n📊 Training Summary:")
        print(f"   Best model saved as: {results.save_dir}/weights/best.pt")
        print(f"   Final model saved as: {results.save_dir}/weights/last.pt")
        print(f"   Results saved in: {results.save_dir}")

if __name__ == "__main__":
    main()
