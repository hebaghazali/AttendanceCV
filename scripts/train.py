"""
Training script for the AttendanceCV system
"""
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from attendance.models.trainer import ModelTrainer
from attendance.utils.config import config
from attendance.utils.logger import logger

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train face recognition models')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--grid-search', action='store_true', help='Use grid search for hyperparameter tuning')
    return parser.parse_args()

def main():
    """Main training function"""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    if args.config:
        config.load_config(args.config)
    else:
        config.load_config()
    
    logger.info("Starting model training...")
    
    # Initialize and run trainer
    trainer = ModelTrainer(use_grid_search=args.grid_search)
    models, n_samples = trainer.run_training_pipeline()
    
    if n_samples > 0 and models:
        logger.info(f"Training completed successfully. Trained on {n_samples} samples.")
        return 0
    else:
        logger.error("Training failed: no samples or models produced")
        return 1
    
if __name__ == "__main__":
    sys.exit(main())