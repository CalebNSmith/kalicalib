import argparse
from pathlib import Path
import json
import random
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import os

def setup_logging(output_dir):
    """Setup logging configuration."""
    log_file = output_dir / f"split_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler()
        ]
    )

def get_image_label_pairs(source_dir):
    """Get all valid image-label pairs."""
    pairs = []
    image_dir = source_dir / 'images'
    label_dir = source_dir / 'labels'
    
    for image_file in image_dir.glob('*.jpg'):
        base_name = image_file.stem
        # Look for matching label file with the same base name
        label_file = label_dir / f"{base_name}.npz"
        if label_file.exists():
            # Extract game name from the filename (assuming format: game-name-number.jpg)
            # This will handle filenames like "charleston_vs_west_liberty-12-11-2024-9_150.jpg"
            game_name = base_name.split('-')[0]
            pairs.append((image_file, label_file, game_name))
        else:
            logging.warning(f"No matching label found for image {image_file}")
    
    return pairs

def create_split(source_dir, output_dir, train_size=0.8, val_size=0.2, test_size=None, seed=42):
    """
    Split dataset into training, validation and test sets using symlinks.
    
    Args:
        source_dir (Path): Directory containing 'images' and 'labels' folders
        output_dir (Path): Directory where split datasets will be saved
        train_size (float): Proportion of data for training
        val_size (float): Proportion of data for validation
        seed (int): Random seed for reproducibility
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    dirs_to_create = [train_dir, val_dir]
    
    # Only create test directory if test_size is specified
    test_dir = None
    if test_size is not None:
        test_dir = output_dir / 'test'
        dirs_to_create.append(test_dir)
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / 'images').mkdir(exist_ok=True)
        (dir_path / 'labels').mkdir(exist_ok=True)
    
    # Get all pairs and their associated game names
    all_pairs = get_image_label_pairs(source_dir)
    
    if not all_pairs:
        raise ValueError("No image-label pairs found")
    
    # Group pairs by game
    games = {}
    for img, lbl, game in all_pairs:
        if game not in games:
            games[game] = []
        games[game].append((img, lbl))
    
    logging.info(f"Found {len(games)} games: {list(games.keys())}")
    
    # Function to create symlinks for a pair
    def create_pair_symlinks(image_file, label_file, target_dir):
        os.symlink(
            image_file.absolute(),
            target_dir / 'images' / image_file.name
        )
        os.symlink(
            label_file.absolute(),
            target_dir / 'labels' / label_file.name
        )
    
    # Initialize empty lists for splits
    train_pairs = []
    val_pairs = []
    test_pairs = []
    
    # Adjust sizes based on whether we're doing train/val only or train/val/test
    # No need to normalize train and val sizes - train_test_split will handle it
    # Just validate that they sum to approximately 1.0 if test_size is None
    if test_size is None and abs(train_size + val_size - 1.0) > 0.001:
        logging.warning(f"train_size ({train_size}) + val_size ({val_size}) != 1.0")
        total = train_size + val_size
        train_size = train_size / total
        val_size = val_size / total
        logging.info(f"Normalized sizes to: train_size={train_size:.3f}, val_size={val_size:.3f}")
    
    for game, pairs in games.items():
        if test_size is None:
            # For train/val split, convert val_size to test_size for sklearn
            train_game, val_game = train_test_split(
                pairs,
                train_size=train_size,
                test_size=val_size,
                random_state=seed
            )
            train_pairs.extend(train_game)
            val_pairs.extend(val_game)
            logging.info(f"{game}: {len(train_game)} train, {len(val_game)} val")
        else:
            # Do train/val/test split
            train_game, temp = train_test_split(
                pairs,
                train_size=train_size,
                random_state=seed
            )
            
            # Then split temp into val and test
            relative_val_size = val_size / (val_size + test_size)
            val_game, test_game = train_test_split(
                temp,
                train_size=relative_val_size,
                random_state=seed
            )
            
            train_pairs.extend(train_game)
            val_pairs.extend(val_game)
            test_pairs.extend(test_game)
            logging.info(f"{game}: {len(train_game)} train, {len(val_game)} val, {len(test_game)} test")
    
    # Create symlinks for each split
    logging.info(f"Creating train split with {len(train_pairs)} pairs")
    for image_file, label_file in train_pairs:
        create_pair_symlinks(image_file, label_file, train_dir)
    
    logging.info(f"Creating validation split with {len(val_pairs)} pairs")
    for image_file, label_file in val_pairs:
        create_pair_symlinks(image_file, label_file, val_dir)
    
    if test_size is not None:
        logging.info(f"Creating test split with {len(test_pairs)} pairs")
        for image_file, label_file in test_pairs:
            create_pair_symlinks(image_file, label_file, test_dir)
    
    # Save split information
    split_info = {
        'num_games': len(games),
        'games': list(games.keys()),
        'num_train_pairs': len(train_pairs),
        'num_val_pairs': len(val_pairs),
        'num_test_pairs': len(test_pairs),
        'split_ratios': {
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size
        }
    }
    
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    logging.info(f"Split information saved to {output_dir / 'split_info.json'}")

def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets using symlinks')
    parser.add_argument('--source-dir', type=str, required=True,
                       help='Path to directory containing images and labels folders')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Path to output directory for split datasets')
    parser.add_argument('--train-size', type=float, default=0.8,
                       help='Proportion of data for training')
    parser.add_argument('--val-size', type=float, default=0.2,
                       help='Proportion of data for validation')
    parser.add_argument('--test-size', type=float, default=None,
                       help='Proportion of data for testing (optional)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    
    # Log arguments
    logging.info(f"Arguments:\n{vars(args)}")
    
    # Validate split sizes
    if args.test_size is not None:
        if args.train_size + args.val_size + args.test_size != 1.0:
            raise ValueError("When using test split, train_size + val_size + test_size must equal 1.0")
    
    # Create splits
    try:
        create_split(
            args.source_dir,
            args.output_dir,
            args.train_size,
            args.val_size,
            args.test_size,  # Pass test_size properly
            args.seed        # Pass seed properly
        )
    except Exception as e:
        logging.error(f"Error during split creation: {str(e)}")
        raise
    
    logging.info("Dataset split completed successfully")

if __name__ == '__main__':
    main()