#!/usr/bin/env python

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# Add the project root to the Python path (so we can import src.*)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ---- Import your modules ----
from src.models.network import KaliCalibNet
from src.training.losses import KeypointsCrossEntropyLoss
from src.data.dataset import KaliCalibDataset
from src.data.data_prep import KaliCalibDataPrep
from src.data.heatmap_transforms import ColorJitter, RandomHorizontalFlip

###############################################################################
# Utility functions
###############################################################################

class ComposeTransforms:
    """Composes multiple transforms together, handling both image and target transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            if hasattr(t, '__call__'):
                # If the transform accepts both image & target, call it that way
                if 'target' in t.__call__.__code__.co_varnames:
                    image, target = t(image, target)
                else:
                    image = t(image)
        
        # Ensure final image is in torch tensor format (CHW)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
        elif isinstance(image, torch.Tensor) and image.dim() == 3 and image.shape[0] not in (1,3):
            # If tensor but in HWC format, convert to CHW
            image = image.permute(2, 0, 1)
        
        return image if target is None else (image, target)

def create_timestamped_dir(base_dir):
    """Create a timestamped directory within the base directory."""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    timestamped_dir = Path(base_dir) / timestamp
    timestamped_dir.mkdir(parents=True, exist_ok=True)
    return timestamped_dir

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Run one epoch of training on the entire training set.
    Returns the average loss over this epoch.
    """
    model.train()
    total_loss = 0.0
    for batch_idx, (images, heatmaps) in enumerate(dataloader):
        images = images.to(device)       # shape (B, 3, H, W)
        heatmaps = heatmaps.to(device)   # shape (B, K+1, H/4, W/4), if output_stride=4

        optimizer.zero_grad()
        outputs = model(images)          # shape (B, K+1, H/4, W/4)
        loss = criterion(outputs, heatmaps)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation set.
    Returns the average validation loss.
    """
    model.eval()
    total_val_loss = 0.0
    for images, heatmaps in dataloader:
        images = images.to(device)
        heatmaps = heatmaps.to(device)

        outputs = model(images)
        loss = criterion(outputs, heatmaps)
        total_val_loss += loss.item()
    
    avg_loss = total_val_loss / len(dataloader)
    return avg_loss

###############################################################################
# Main
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train KaliCalibNet on full dataset")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to a YAML config file (e.g. configs/default.yaml)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Root directory containing train/ and val/ subfolders with images/ and labels/')
    parser.add_argument('--output-dir', type=str, default='outputs/train_runs',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of DataLoader workers')

    # -------------------------------------------------------------------------
    # Optional overrides for YAML values:
    # (Feel free to add or remove any as needed)
    # -------------------------------------------------------------------------
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override training.batch_size in YAML')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Override training.learning_rate in YAML')
    parser.add_argument('--n-epochs', type=int, default=None,
                        help='Override training.n_epochs in YAML')
    parser.add_argument('--lr-decay-epoch', type=int, default=None,
                        help='Override training.lr_decay_epoch in YAML')
    parser.add_argument('--keypoint-weight', type=float, default=None,
                        help='Override training.keypoint_weight in YAML')
    parser.add_argument('--background-weight', type=float, default=None,
                        help='Override training.background_weight in YAML')

    return parser.parse_args()

def create_transforms(config, split='train'):
    """Create transform pipeline based on config and split type."""
    if split == 'train':
        # Get augmentation config with defaults
        aug_config = config.get('augmentation', {})
        color_config = aug_config.get('color_jitter', {})

        transforms_list = [
            ColorJitter(
                brightness=color_config.get('brightness', 0.7),
                contrast=color_config.get('contrast', 0.5),
                saturation=color_config.get('saturation', 0.5),
                hue=color_config.get('hue', 0.5)
            ),
            RandomHorizontalFlip(
                p=aug_config.get('random_flip_prob', 0.5)
            )
        ]
        return ComposeTransforms(transforms_list)
    else:
        # No augmentations for validation
        return None

def main():
    args = parse_args()

    # ---------------------------------------------------------------------
    # 1. Create timestamped output directory
    # ---------------------------------------------------------------------
    run_dir = create_timestamped_dir(args.output_dir)
    
    # ---------------------------------------------------------------------
    # 2. Load config from YAML
    # ---------------------------------------------------------------------
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # ---------------------------------------------------------------------
    # 3. Override YAML values with CLI arguments (if provided)
    # ---------------------------------------------------------------------
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate

    if args.n_epochs is not None:
        config['training']['n_epochs'] = args.n_epochs

    if args.lr_decay_epoch is not None:
        config['training']['lr_decay_epoch'] = args.lr_decay_epoch

    if args.keypoint_weight is not None:
        config['training']['keypoint_weight'] = args.keypoint_weight

    if args.background_weight is not None:
        config['training']['background_weight'] = args.background_weight

    # ---------------------------------------------------------------------
    # 4. Initialize Logging
    # ---------------------------------------------------------------------
    logging.basicConfig(
        filename=str(run_dir / 'training.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Print final config at the start of training
    logging.info("============  TRAINING START  ============")
    logging.info(f"Loaded base config from {args.config}")
    logging.info(f"Data root: {args.data_dir}")
    logging.info(f"Outputs will be saved to {run_dir}")
    logging.info("==== Final Merged Configuration ====")
    logging.info("\n" + yaml.dump(config, sort_keys=False))
    logging.info("====================================")

    # ---------------------------------------------------------------------
    # 5. Prepare Data (train + val)
    # ---------------------------------------------------------------------
    data_prep = KaliCalibDataPrep(config)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_transforms = create_transforms(config, split='train')
    val_transforms = create_transforms(config, split='val')

    train_dataset = KaliCalibDataset(
        data_dir=train_dir,
        data_prep=data_prep,
        transform=train_transforms,
        split='train'
    )
    val_dataset = KaliCalibDataset(
        data_dir=val_dir,
        data_prep=data_prep,
        transform=val_transforms,
        split='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training'].get('batch_size', 2),
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].get('batch_size', 2),
        shuffle=False,
        num_workers=args.num_workers
    )

    # ---------------------------------------------------------------------
    # 6. Initialize Model
    # ---------------------------------------------------------------------
    n_keypoints = config['model']['n_keypoints']  # e.g. 93
    model = KaliCalibNet(n_keypoints)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logging.info("Initialized KaliCalibNet model.")

    # ---------------------------------------------------------------------
    # 7. Create Loss & Optimizer
    # ---------------------------------------------------------------------
    key_wt = config['training'].get('keypoint_weight', 50)
    bg_wt = config['training'].get('background_weight', 1)

    weights = torch.ones(n_keypoints + 1, device=device) * key_wt
    weights[-1] = bg_wt  # last channel is background
    criterion = KeypointsCrossEntropyLoss(weights=weights)

    learning_rate = config['training'].get('learning_rate', 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    lr_decay_epoch = config['training'].get('lr_decay_epoch', None)

    # ---------------------------------------------------------------------
    # 8. Training Loop
    # ---------------------------------------------------------------------
    n_epochs = config['training'].get('n_epochs', 100)
    logging.info(f"Starting training for {n_epochs} epochs.")

    best_val_loss = float('inf')
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        if lr_decay_epoch is not None and epoch == lr_decay_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            logging.info(f"LR decayed to {optimizer.param_groups[0]['lr']} at epoch {epoch}")

        val_loss = evaluate(model, val_loader, criterion, device)

        logging.info(f"[Epoch {epoch:03d}/{n_epochs}] "
                     f"Train Loss: {train_loss:.6f} | "
                     f"Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = run_dir / 'best_model.pth'
            torch.save(model.state_dict(), save_path)
            logging.info(f"  ** New best val loss. Model saved to {save_path}")

    logging.info("Training complete. Best val loss = {:.6f}".format(best_val_loss))
    logging.info("============  TRAINING END  ============")

if __name__ == "__main__":
    main()