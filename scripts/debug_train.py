import os
import sys
import argparse
from pathlib import Path
import yaml
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.trainer import KaliCalibTrainer

def setup_logging(output_dir):
    """Setup basic logging configuration."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / f"training.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler()
        ]
    )
    return run_dir

def main():
    parser = argparse.ArgumentParser(description='Train KaliCalib model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to data directory containing train/val/test splits')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Path to output directory')
    args = parser.parse_args()

    # Create base output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup logging in a timestamped subdirectory
    run_output_dir = setup_logging(output_path)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    logging.info(f"\nConfiguration:\n{yaml.dump(config)}")

    # Initialize trainer and start training
    try:
        trainer = KaliCalibTrainer(config)
        logging.info(f"Using device: {trainer.device}")
        trainer.train(args.data_dir)
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise

    logging.info("Training script finished")

if __name__ == '__main__':
    main()