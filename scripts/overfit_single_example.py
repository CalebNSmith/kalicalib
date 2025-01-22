import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path (so we can import src.*)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
import cv2

from src.models.network import KaliCalibNet
from src.training.losses import KeypointsCrossEntropyLoss
from src.data.dataset import KaliCalibDataset
from src.data.data_prep import KaliCalibDataPrep
from src.data.heatmap_transforms import RandomHorizontalFlip

def visualize_keypoints(image, heatmaps, threshold=0.5):
    """Visualize keypoints and bounds on the image."""
    img = image.cpu().numpy().transpose(1, 2, 0)
    if img.max() > 1.0:
        img = img / 255.0
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    
    heatmaps = heatmaps.cpu().numpy()
    
    # Plot grid points (indices 0-90)
    for k in range(91):
        hmap = heatmaps[k]
        if hmap.max() > threshold:
            y, x = np.unravel_index(hmap.argmax(), hmap.shape)
            x = x * 4
            y = y * 4
            plt.plot(x, y, 'r.', markersize=10)
            plt.text(x+5, y+5, f'grid_{k}', color='white', 
                    bbox=dict(facecolor='red', alpha=0.5))
    
    # Plot upper bound points (index 91)
    hmap = heatmaps[91]
    if hmap.max() > threshold:
        y, x = np.unravel_index(hmap.argmax(), hmap.shape)
        x = x * 4
        y = y * 4
        plt.plot(x, y, 'b.', markersize=10)
        plt.text(x+5, y+5, 'ub', color='white',
                bbox=dict(facecolor='blue', alpha=0.5))

    # Plot lower bound points (index 92)
    hmap = heatmaps[92]
    if hmap.max() > threshold:
        y, x = np.unravel_index(hmap.argmax(), hmap.shape)
        x = x * 4
        y = y * 4
        plt.plot(x, y, 'g.', markersize=10)
        plt.text(x+5, y+5, 'lb', color='white',
                bbox=dict(facecolor='green', alpha=0.5))
    
    plt.axis('off')
    return plt.gcf()

def save_inference_visualization(model, image, heatmap, output_dir, suffix=''):
    """
    Run inference on image and save visualization.
    
    Args:
        model: trained KaliCalibNet model
        image: input image tensor of shape (1, 3, H, W)
        heatmap: ground truth heatmap
        output_dir: directory to save visualization
        suffix: optional suffix for the output filename
    """
    print(f"\nModel Output Analysis{suffix}:")
    print("-" * 50)
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        prob_output = model(image)[0]
        output = prob_output
        
        # Print raw output statistics
        print(f"Raw output shape: {prob_output.shape}")
        print(f"Raw output range: [{prob_output.min():.3f}, {prob_output.max():.3f}]")
        print(f"Raw output mean: {prob_output.mean():.3f}")
        print(f"Raw output std: {prob_output.std():.3f}")
        
        # Print confidence scores for detected keypoints
        confidences = []
        for k in range(1, output.shape[0]):  # Skip background channel
            conf = output[k].max().item()
            if conf > 0.5:  # Same threshold as visualization
                confidences.append((k, conf))
        
        print("\nDetected Keypoints (conf > 0.5):")
        for k, conf in sorted(confidences, key=lambda x: x[1], reverse=True):
            print(f"Keypoint {k}: confidence = {conf:.3f}")
            y, x = np.unravel_index(output[k].cpu().argmax(), output[k].shape)
            print(f"    Position (x, y): ({x*4}, {y*4}) in original image coordinates")
    
    # Create visualization
    fig = visualize_keypoints(image[0], output)
    
    # Save visualization
    filename = f'overfit_result{suffix}.png'
    fig.savefig(output_dir / filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Also visualize ground truth
    fig = visualize_keypoints(image[0], heatmap[0])
    filename = f'ground_truth{suffix}.png'
    fig.savefig(output_dir / filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    print(f"Saved visualizations to {output_dir}")



def main():
    parser = argparse.ArgumentParser(description="Overfit a single example for debugging")
    parser.add_argument('--config', type=str, default='configs/default.yaml', 
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data directory containing images/labels')
    parser.add_argument('--iterations', type=int, default=200,
                        help='Number of iterations to run on the single example')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for optimizer')
    parser.add_argument('--flip', action='store_true',
                        help='Apply horizontal flip to the training example')
    args = parser.parse_args()

    # 1. Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Prepare data prep and dataset
    data_prep = KaliCalibDataPrep(config)
    dataset = KaliCalibDataset(
        data_dir=args.data_dir,
        data_prep=data_prep,
        transform=None,
        split='train'
    )

    # Get first sample
    image, heatmap = dataset[0]
    print("\nInput Data Analysis:")
    print("-" * 50)
    print(f"Image shape: {image.shape}")
    print(f"Label heatmap shape: {heatmap.shape}")
    
    # Apply horizontal flip if requested
    if args.flip:
        print("\nApplying horizontal flip to training example...")
        # Flip image
        image = torch.flip(image, [2])  # Flip along width dimension
        
        # Flip heatmap and remap keypoints
        heatmap = torch.flip(heatmap, [2])  # First flip spatially
        
        # Create a new heatmap tensor for remapped points
        remapped_heatmap = torch.zeros_like(heatmap)
        
        # Remap grid points (0-90)
        # Assuming 13x7 grid (91 points), with points ordered left-to-right, top-to-bottom
        grid_width = 13
        grid_height = 7
        for y in range(grid_height):
            for x in range(grid_width):
                old_idx = y * grid_width + x
                new_x = grid_width - 1 - x  # Flip x coordinate
                new_idx = y * grid_width + new_x
                remapped_heatmap[new_idx] = heatmap[old_idx]
        
        # Remap basket points (indices 91 and 92)
        # Upper basket point (91) and lower basket point (92) stay at same indices
        remapped_heatmap[91] = heatmap[91]  # ub
        remapped_heatmap[92] = heatmap[92]  # lb
        
        # Background channel stays the same
        remapped_heatmap[93] = heatmap[93]
        
        heatmap = remapped_heatmap
        print("Applied horizontal flip with keypoint remapping")
        
    # 3. Move everything to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.unsqueeze(0).to(device)    # (1, 3, H, W)
    heatmap = heatmap.unsqueeze(0).to(device)  # (1, K+1, H/4, W/4)

    # 4. Initialize the model
    n_keypoints = config['model']['n_keypoints']
    model = KaliCalibNet(n_keypoints).to(device)
    model.train()

    # 5. Create the loss function
    key_wt = config['training'].get('keypoint_weight', 1000)
    bg_wt = config['training'].get('background_weight', 1)
    weights = torch.ones(n_keypoints + 1, device=device) * key_wt
    weights[-1] = bg_wt
    criterion = KeypointsCrossEntropyLoss(weights=weights)

    # 6. Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 7. Overfit loop
    print("[DEBUG] Starting overfit loop on a single sample...")
    for i in range(args.iterations):
        optimizer.zero_grad()
        out = model(image)
        loss = criterion(out, heatmap)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{args.iterations}, Loss = {loss.item():.6f}")

    print("Overfit debugging run finished!")

    # 8. Save inference visualization
    suffix = '_flipped' if args.flip else ''
    save_inference_visualization(model, image, heatmap, 'outputs/overfit', suffix)

if __name__ == "__main__":
    main()