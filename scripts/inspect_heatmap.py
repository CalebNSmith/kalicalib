#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import argparse

def get_grid_points():
    """Get list of grid point names in order."""
    # Grid size from KaliCalibDataPrep
    rows, cols = 7, 13
    grid_points = []
    
    # Generate grid point names (row by row)
    for i in range(rows):
        for j in range(cols):
            grid_points.append(f"grid_{i}_{j}")
            
    return grid_points

def inspect_heatmaps(npz_path, image_path=None, save_dir=None, num_channels=16):
    """
    Inspect a heatmap NPZ file and visualize its contents.
    
    Args:
        npz_path: Path to the NPZ file
        image_path: Optional path to corresponding image
        save_dir: Optional directory to save visualizations
        num_channels: Number of channels to display
    """
    # Load the NPZ file
    data = np.load(npz_path)
    heatmaps = data['heatmaps']
    
    # Get grid points
    grid_points = get_grid_points()
    
    print(f"\nHeatmap file: {npz_path}")
    print(f"Shape: {heatmaps.shape}")
    print(f"Number of channels: {heatmaps.shape[0]} (1 background + {heatmaps.shape[0]-1} keypoints)")
    print(f"Resolution: {heatmaps.shape[1:]} (1/4 of input size)")
    print(f"Min value: {heatmaps.min():.6f}")
    print(f"Max value: {heatmaps.max():.6f}")
    print(f"Mean value: {heatmaps.mean():.6f}")
    
    # Analyze each channel
    print("\nChannel analysis:")
    print("Channel 0: Background")
    for i in range(1, heatmaps.shape[0]):
        channel = heatmaps[i]
        point_name = grid_points[i-1] if i-1 < len(grid_points) else f"point_{i}"
        max_pos = np.unravel_index(channel.argmax(), channel.shape)
        print(f"Channel {i}: {point_name} - Peak at {max_pos} (value: {channel.max():.3f})")
    
    # Create visualization grid
    n_channels = min(num_channels, heatmaps.shape[0])
    rows = int(np.ceil(np.sqrt(n_channels)))
    cols = int(np.ceil(n_channels / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    
    # Plot each channel
    for idx in range(n_channels):
        row = idx // cols
        col = idx % cols
        
        # Get heatmap and point name
        heatmap = heatmaps[idx]
        point_name = "background" if idx == 0 else grid_points[idx-1] if idx-1 < len(grid_points) else f"point_{idx}"
        
        # Plot heatmap
        im = axes[row, col].imshow(heatmap, cmap='viridis')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Channel {idx}: {point_name}')
        plt.colorbar(im, ax=axes[row, col])
        
        # Add marker at peak location
        if idx > 0:  # Skip background
            max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
            axes[row, col].plot(max_pos[1], max_pos[0], 'r+', markersize=10)
    
    # Hide empty subplots
    for idx in range(n_channels, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # If original image is provided, create comparison visualization
    if image_path:
        img = cv2.imread(str(image_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(20, 5))
            
            # Original image
            plt.subplot(131)
            plt.imshow(img)
            plt.title('Original Image')
            plt.axis('off')
            
            # Combined heatmap (skip background)
            plt.subplot(132)
            combined_heatmap = np.max(heatmaps[1:], axis=0)
            plt.imshow(combined_heatmap, cmap='viridis')
            plt.title('Combined Keypoint Heatmaps')
            plt.colorbar()
            plt.axis('off')
            
            # Background channel
            plt.subplot(133)
            plt.imshow(heatmaps[0], cmap='viridis')
            plt.title('Background Channel')
            plt.colorbar()
            plt.axis('off')
            
            plt.tight_layout()
    
    # Save or show
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save basic visualization
        save_path = save_dir / f"{Path(npz_path).stem}_analysis.png"
        plt.savefig(save_path)
        print(f"\nSaved visualization to: {save_path}")
        
        # Optionally save individual channel heatmaps
        heatmap_dir = save_dir / Path(npz_path).stem
        heatmap_dir.mkdir(exist_ok=True)
        
        for i in range(heatmaps.shape[0]):
            plt.figure(figsize=(8, 8))
            plt.imshow(heatmaps[i], cmap='viridis')
            plt.colorbar()
            point_name = "background" if i == 0 else grid_points[i-1] if i-1 < len(grid_points) else f"point_{i}"
            plt.title(f'Channel {i}: {point_name}')
            plt.axis('off')
            plt.savefig(heatmap_dir / f"channel_{i:03d}_{point_name}.png")
            plt.close()
        
        print(f"Saved individual channel visualizations to: {heatmap_dir}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Inspect heatmap NPZ files')
    parser.add_argument('npz_path', type=str, help='Path to NPZ file')
    parser.add_argument('--image', type=str, help='Path to corresponding image file')
    parser.add_argument('--save-dir', type=str, help='Directory to save visualizations')
    parser.add_argument('--channels', type=int, default=16, help='Number of channels to display')
    args = parser.parse_args()
    
    inspect_heatmaps(args.npz_path, args.image, args.save_dir, args.channels)

if __name__ == '__main__':
    main()