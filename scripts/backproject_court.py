#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# Adjust Python path to import your code if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.network import KaliCalibNet
from src.data.data_prep import KaliCalibDataPrep
# If you have any "transform_point" logic from src/utils/court.py, you can import that too
# from src.utils.court import transform_point

def load_image(image_path, data_prep):
    """
    Load and preprocess a single image for the network (resizing to input_size, etc.).
    Returns: (image_tensor, original_img_for_display)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to load: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to match model input if necessary
    if image.shape[:2] != (data_prep.input_size[1], data_prep.input_size[0]):
        image = cv2.resize(image, data_prep.input_size)
    
    # Keep a copy for visualization
    display_img = image.copy()
    
    # Transform to tensor the same way your pipeline expects
    image_tensor = data_prep.transform(image)  # shape (3, H, W)
    image_tensor = image_tensor.unsqueeze(0)   # shape (1, 3, H, W)
    return image_tensor, display_img

def find_keypoints_from_heatmaps(heatmaps, threshold=0.5, stride=4):
    """
    Given raw heatmaps of shape (K+1, H, W),
    find the (x, y) location in *image coordinates* for each channel
    whose peak > threshold.
    
    Returns a dict: { channel_idx: (x, y) } in the *image* reference frame.
    """
    # heatmaps: torch.Tensor or np.ndarray of shape (K+1, H, W)
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()

    num_channels, h, w = heatmaps.shape
    keypoints = {}  # channel -> (x_image, y_image)

    for ch in range(num_channels):
        channel_hm = heatmaps[ch]
        # Find the maximum activation in this channel
        val = channel_hm.max()
        if val > threshold:
            # Argmax to get location
            idx = channel_hm.argmax()
            y, x = np.unravel_index(idx, channel_hm.shape)
            # Scale back up to original network input size
            # because the heatmap is smaller by 'stride'
            x_img = x * stride
            y_img = y * stride
            keypoints[ch] = (x_img, y_img)
        else:
            keypoints[ch] = None  # No confident detection
    return keypoints

def invert_points(keypoints, homography):
    """
    Given a dict of {ch: (x_img, y_img)} in image space,
    and a homography that maps court -> image,
    we compute the inverse to go image -> court.

    Returns a dict {ch: (X_court, Y_court)}.
    """
    H_inv = np.linalg.inv(homography)

    out_dict = {}
    for ch, pt in keypoints.items():
        if pt is None:
            out_dict[ch] = None
            continue
        x_img, y_img = pt

        # Convert to homogeneous coords
        p = np.array([x_img, y_img, 1.0], dtype=np.float64)
        p_court = H_inv @ p
        # Convert back from homogeneous
        X_court = p_court[0] / p_court[2]
        Y_court = p_court[1] / p_court[2]
        out_dict[ch] = (X_court, Y_court)
    return out_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Path to .pth model weights")
    parser.add_argument('--image', type=str, required=True, help="Path to input image")
    parser.add_argument('--homography', type=str, required=True, 
                        help="Path to a .npz file containing 'H' (3x3) that maps court->image")
    parser.add_argument('--config', type=str, default=None, help="Path to config yaml (optional)")
    parser.add_argument('--threshold', type=float, default=0.5, help="Heatmap peak threshold")
    parser.add_argument('--stride', type=int, default=4, 
                        help="Network output stride used for upscaling the predicted heatmaps back to image size")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Directory to save visualization outputs")
    args = parser.parse_args()

    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load config if available
    if args.config is not None:
        import yaml
        from src.data.data_prep import KaliCalibDataPrep
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        data_prep = KaliCalibDataPrep(config)
    else:
        # Minimal fallback if no config is provided:
        from src.data.data_prep import KaliCalibDataPrep
        config = {
            'model': {
                'input_size': [960, 540]  # example defaults
            }
        }
        data_prep = KaliCalibDataPrep(config)

    # 2) Load homography from npz
    data = np.load(args.homography)
    if 'H' not in data:
        raise KeyError(f"NPZ file {args.homography} is missing 'H' array.")
    H = data['H']  # shape (3, 3)

    # 3) Load model
    n_keypoints = 93  # or from config['model']['n_keypoints']
    model = KaliCalibNet(n_keypoints=n_keypoints)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    # 4) Load + preprocess image
    image_tensor, display_img = load_image(args.image, data_prep)
    image_tensor = image_tensor.to(device)

    # 5) Inference
    with torch.no_grad():
        # Output shape: (1, K+1, H/stride, W/stride)
        heatmaps = model(image_tensor)[0]  # shape (K+1, H/stride, W/stride)

    # 6) Extract (x_img, y_img) for each channel
    keypoints_image = find_keypoints_from_heatmaps(
        heatmaps, threshold=args.threshold, stride=args.stride
    )

    # 7) Invert to court coords
    keypoints_court = invert_points(keypoints_image, H)

    # 8) Print results
    print("=== PREDICTED KEYPOINTS (IMAGE SPACE) ===")
    for ch, pt in keypoints_image.items():
        print(f"Channel {ch}: {pt}")

    print("\n=== PREDICTED KEYPOINTS (COURT SPACE) ===")
    for ch, pt in keypoints_court.items():
        print(f"Channel {ch}: {pt}")

    # 9) Visualization
    # First figure: Raw image space predictions
    fig1 = plt.figure(figsize=(16, 9))
    ax1 = fig1.add_subplot(111)
    ax1.imshow(display_img)
    # Use the raw image space coordinates
    for ch, pt in keypoints_image.items():
        if pt is not None:
            x, y = pt[0], pt[1]  # These are already in image space
            ax1.plot(x, y, 'go', markersize=8)
            ax1.text(x+5, y+5, str(ch), color='white', bbox=dict(facecolor='black', alpha=0.7))
    ax1.set_title("Raw Predicted Image Space Points")
    ax1.axis('off')
    plt.tight_layout()
    
    if args.output_dir:
        # Save first figure
        base_filename = os.path.splitext(os.path.basename(args.image))[0]
        output_path1 = os.path.join(args.output_dir, f"{base_filename}_raw_predictions.png")
        plt.savefig(output_path1, dpi=300, bbox_inches='tight')
        print(f"Saved raw predictions visualization to: {output_path1}")
    plt.close()

    # Second figure: Court space coordinates
    fig2 = plt.figure(figsize=(16, 9))
    ax2 = fig2.add_subplot(111)
    ax2.set_title("Court Space Points")
    # Plot the actual court space coordinates
    court_points = np.array([(pt[0], pt[1]) for pt in keypoints_court.values() if pt is not None])
    if len(court_points) > 0:
        ax2.scatter(court_points[:, 0], court_points[:, 1], c='red', s=50)
        for ch, pt in keypoints_court.items():
            if pt is not None:
                ax2.text(pt[0], pt[1], str(ch), fontsize=8)
    ax2.grid(True)
    ax2.set_aspect('equal')
    plt.tight_layout()

    if args.output_dir:
        # Save second figure
        output_path2 = os.path.join(args.output_dir, f"{base_filename}_court_space.png")
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        print(f"Saved court space visualization to: {output_path2}")
    plt.close()
    if args.output_dir:
        # Extract filename from input image path
        base_filename = os.path.splitext(os.path.basename(args.image))[0]
        output_path = os.path.join(args.output_dir, f"{base_filename}_keypoints.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    main()
