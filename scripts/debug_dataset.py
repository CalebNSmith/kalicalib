import argparse
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os, sys
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.dataset import KaliCalibDataset
from src.data.data_prep import KaliCalibDataPrep

def validate_onehot(grid_maps, stage_name=""):
    """
    Validate that labels form a proper one-hot encoding where each pixel 
    has exactly one active channel.
    
    Args:
        grid_maps: List of grid maps of shape (H, W) or single array of shape (C, H, W)
        stage_name: Name of the validation stage for reporting
    """
    print(f"\nValidating one-hot encoding ({stage_name}):")
    
    # Convert list of grids to a single array if needed
    if isinstance(grid_maps, list):
        # Stack along channel dimension
        grid_maps = np.stack(grid_maps, axis=0)
    
    H, W = grid_maps.shape[-2:]
    total_pixels = H * W
    
    # Sum across channels for each pixel
    channel_sum = np.sum(grid_maps, axis=0)
    
    # Check conditions
    zeros = np.isclose(channel_sum, 0)
    ones = np.isclose(channel_sum, 1)
    others = ~(zeros | ones)
    
    n_zeros = np.sum(zeros)
    n_ones = np.sum(ones)
    n_others = np.sum(others)
    
    print(f"Total pixels: {total_pixels}")
    print(f"Pixels with no active channel: {n_zeros} ({n_zeros/total_pixels*100:.2f}%)")
    print(f"Pixels with exactly one active channel: {n_ones} ({n_ones/total_pixels*100:.2f}%)")
    print(f"Pixels with multiple active channels: {n_others} ({n_others/total_pixels*100:.2f}%)")
    
    if n_others > 0:
        print("⚠️ Warning: Found pixels with multiple active channels!")
        # Find example problematic pixels
        problem_y, problem_x = np.where(others)[:5]  # Get up to 5 examples
        print("Example problematic pixels (y, x, sum):")
        for y, x in zip(problem_y, problem_x):
            print(f"  Position ({y}, {x}): sum = {channel_sum[y, x]:.6f}")
            active_channels = np.where(grid_maps[:, y, x] > 0)[0]
            print(f"    Active channels: {active_channels}")
            print(f"    Channel values: {grid_maps[:, y, x][active_channels]}")
    
    if n_zeros == total_pixels:
        print("⚠️ Warning: All pixels are zero!")
    
    return {
        'valid': n_ones == total_pixels,
        'zeros': n_zeros,
        'ones': n_ones,
        'others': n_others
    }

def find_peak_locations(label, threshold=0.5):
    """Find peak locations in a label map."""
    peaks = []
    if label.max() > 0:  # Normalize if values exceed 1
        label = label / label.max()
    mask = label > threshold
    coordinates = np.where(mask)
    for y, x in zip(*coordinates):
        peaks.append((x, y))  # Return as (x,y) for easier plotting
    return peaks

def debug_pipeline_visualization(image_path, label_path, data_prep, output_dir, split="unknown"):
    """Create visualizations for validating the label downsampling process."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    raw_image = cv2.imread(str(image_path))
    print(f"\nStage 1: Raw Data")
    print(f"Raw image shape: {raw_image.shape}")
    
    with np.load(label_path, allow_pickle=True) as npz_data:
        print("\nLabel file contains:")
        grid_maps = []
        for key in sorted(npz_data.keys()):
            array = npz_data[key]
            print(f"  {key}: shape={array.shape}, range=[{array.min():.6f}, {array.max():.6f}]")
            if key.startswith('grid_'):
                grid_maps.append(array)
        
        # Validate one-hot encoding for all grid maps
        validate_onehot(grid_maps, "Original Resolution")
        
        # For visualization, just use first 3 grids
        grids = grid_maps[:3]
        bg = npz_data['background'] if 'background' in npz_data else None

    # 2. Process each resolution stage
    input_h, input_w = data_prep.input_size[1], data_prep.input_size[0]
    output_h, output_w = input_h // data_prep.output_stride, input_w // data_prep.output_stride

    # Create multi-panel visualization
    fig = plt.figure(figsize=(20, 10))
    gs = plt.GridSpec(3, 4, figure=fig)

    # Row 1: Original Resolution
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.set_title(f"Raw Image\n{raw_image.shape[1]}x{raw_image.shape[0]}")
    ax_img.imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
    
    # Plot first 3 grid points in different colors
    colors = ['red', 'green', 'blue']
    for grid_idx, (grid, color) in enumerate(zip(grids, colors)):
        validate_label_values(grid, f"Grid {grid_idx} (Original)")
        peaks = find_peak_locations(grid)
        if peaks:
            x, y = zip(*peaks)
            ax_img.scatter(x, y, c=color, s=50, label=f'Grid {grid_idx}')
    ax_img.legend()
    ax_img.axis('off')

    # Show individual grid points
    for idx, (grid, color) in enumerate(zip(grids, colors)):
        ax = fig.add_subplot(gs[0, idx+1])
        ax.set_title(f"Grid {idx} (Original)")
        ax.imshow(grid, cmap='viridis')
        peaks = find_peak_locations(grid)
        if peaks:
            x, y = zip(*peaks)
            ax.scatter(x, y, c=color, s=50)
        ax.axis('off')

    # Row 2: Output Resolution (1/4)
    resized_image = cv2.resize(raw_image, (output_w, output_h))
    ax_img_small = fig.add_subplot(gs[1, 0])
    ax_img_small.set_title(f"Resized Image\n{output_w}x{output_h}")
    ax_img_small.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    ax_img_small.axis('off')

    # Show downsampled grid points
    for idx, (grid, color) in enumerate(zip(grids, colors)):
        # Downsample grid
        grid_small = cv2.resize(grid, (output_w, output_h), interpolation=cv2.INTER_AREA)
        validate_label_values(grid_small, f"Grid {idx} (Downsampled)")
        
        ax = fig.add_subplot(gs[1, idx+1])
        ax.set_title(f"Grid {idx} (1/4 Resolution)")
        ax.imshow(grid_small, cmap='viridis')
        
        # Plot peaks in downsampled grid
        peaks = find_peak_locations(grid_small)
        if peaks:
            x, y = zip(*peaks)
            ax = plt.gca()
            ax.scatter(x, y, c=color, s=50)
            # Also plot on resized image
            ax_img_small.scatter(x, y, c=color, s=50, label=f'Grid {idx}')
        ax.axis('off')
    ax_img_small.legend()

    # Row 3: Difference visualization
    ax_diff = fig.add_subplot(gs[2, 0])
    ax_diff.set_title("Peak Locations\nOriginal (x) vs Downsampled (o)")
    ax_diff.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    
    for idx, (grid, color) in enumerate(zip(grids, colors)):
        # Original peaks (scaled to output size)
        peaks_orig = find_peak_locations(grid)
        if peaks_orig:
            x_orig, y_orig = zip(*peaks_orig)
            x_orig = [x * output_w / grid.shape[1] for x in x_orig]
            y_orig = [y * output_h / grid.shape[0] for y in y_orig]
            ax_diff.scatter(x_orig, y_orig, marker='x', c=color, s=100, label=f'Grid {idx} Original')
        
        # Downsampled peaks
        grid_small = cv2.resize(grid, (output_w, output_h), interpolation=cv2.INTER_AREA)
        peaks_small = find_peak_locations(grid_small)
        if peaks_small:
            x_small, y_small = zip(*peaks_small)
            ax_diff.scatter(x_small, y_small, marker='o', facecolors='none', 
                          edgecolors=color, s=100, label=f'Grid {idx} Downsampled')
    ax_diff.legend()
    ax_diff.axis('off')

    # Add analysis of peak shifts
    for idx, (grid, color) in enumerate(zip(grids, colors)):
        peaks_orig = find_peak_locations(grid)
        grid_small = cv2.resize(grid, (output_w, output_h), interpolation=cv2.INTER_AREA)
        peaks_small = find_peak_locations(grid_small)
        
        if peaks_orig and peaks_small:
            # Scale original peaks to output size
            peaks_orig_scaled = [(x * output_w / grid.shape[1], y * output_h / grid.shape[0]) 
                               for x, y in peaks_orig]
            
            # Calculate distances between original and downsampled peaks
            distances = []
            for (x1, y1), (x2, y2) in zip(peaks_orig_scaled, peaks_small):
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                distances.append(dist)
            
            ax = fig.add_subplot(gs[2, idx+1])
            ax.set_title(f"Grid {idx} Peak Shift\nMax: {max(distances):.2f} px")
            ax.hist(distances, bins=20)
            ax.set_xlabel('Distance (pixels)')
            ax.set_ylabel('Count')

    plt.tight_layout()
    
    # Save the visualization
    output_path = output_dir / f"downsample_validation_{split}_{timestamp}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--data_dir", required=True, help="Path to the root dataset directory")
    parser.add_argument("--split", default="train", help="Which dataset split to debug (train/val/test)")
    parser.add_argument("--output_dir", default="outputs/dataset_debug", help="Directory to save debug visualizations")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to visualize")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    data_prep = KaliCalibDataPrep(config)
    dataset = KaliCalibDataset(
        data_dir=args.data_dir,
        data_prep=data_prep,
        transform=None,
        split=args.split
    )

    for i in range(min(args.num_samples, len(dataset))):
        image_path, label_path = dataset.samples[i]
        print(f"\nProcessing sample {i+1}/{args.num_samples}")
        print(f"Image: {image_path}")
        print(f"Label: {label_path}")

        output_path = debug_pipeline_visualization(
            image_path=image_path,
            label_path=label_path,
            data_prep=data_prep,
            output_dir=args.output_dir,
            split=args.split
        )
        print(f"Saved validation visualization to: {output_path}")

if __name__ == "__main__":
    main()