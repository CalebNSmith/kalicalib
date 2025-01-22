#!/usr/bin/env python
"""
Updated inference script using:
  - generate_perspective_aware_grid_points for 91 grid points
  - 'ub'/'lb' basket coordinates as in your generate_binary_heatmaps code
  - Homography from predicted -> court
  - Random point demonstration
"""

import os
import sys
import argparse
import random
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# Adjust these if needed:
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.network import KaliCalibNet
from src.data.data_prep import KaliCalibDataPrep
from src.utils.court import generate_perspective_aware_grid_points, calculate_court_points
# (Optional) from src.utils.court import calculate_court_points, transform_point

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def extract_predicted_points(prediction_tensor, stride=4, num_keypoints=93):
    """
    Convert model's heatmap output (C,H_out,W_out) to (num_keypoints,2) array of (x,y),
    by taking the argmax in each channel & multiplying by stride.
    
    Args:
        prediction_tensor: Model output of shape (C,H_out,W_out)
        stride: Output stride of the model
        num_keypoints: Number of keypoints to extract (excludes background class if present)
    """
    if hasattr(prediction_tensor, 'cpu'):
        prediction_tensor = prediction_tensor.cpu().numpy()
        
    # Only take the first num_keypoints channels (excluding background)
    prediction_tensor = prediction_tensor[:num_keypoints]
    
    channels, h_out, w_out = prediction_tensor.shape
    coords = np.zeros((channels, 2), dtype=np.float32)
    
    for i in range(channels):
        heatmap = prediction_tensor[i]
        max_idx = np.argmax(heatmap)
        row, col = divmod(max_idx, w_out)
        # Scale back to original image coords
        px = col * stride
        py = row * stride
        coords[i] = (px, py)
    
    return coords

def visualize_keypoints(image, keypoints, stride=4, threshold=0.5, title="Predicted Keypoints"):
    """
    Show each channel's argmax on the image via matplotlib:
    - image: (H,W,3) in RGB or float
    - keypoints: shape (93, H_out, W_out)
    - stride: how much the output is downsampled from the input
    - threshold: skip channels where max < threshold
    """
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().numpy().transpose(1,2,0)
    else:
        img = image.copy()

    if img.dtype != np.float32 and img.max() > 1.0:
        img = img.astype(np.float32) / 255.0

    plt.figure(figsize=(12,8))
    plt.imshow(img)
    plt.title(title)

    if hasattr(keypoints, 'cpu'):
        keypoints = keypoints.cpu().numpy()

    # 0..90 => grid, 91 => ub, 92 => lb
    for c in range(keypoints.shape[0]):
        channel_map = keypoints[c]
        if channel_map.max() < threshold:
            continue
        max_idx = channel_map.argmax()
        h_out, w_out = channel_map.shape
        row, col = divmod(max_idx, w_out)
        x = col * stride
        y = row * stride

        if c < 91:
            plt.plot(x, y, 'r.', markersize=8)
            plt.text(x+3, y+3, f'g{c}', color='white',
                     bbox=dict(facecolor='red', alpha=0.5))
        elif c == 91:
            plt.plot(x, y, 'b.', markersize=10)
            plt.text(x+3, y+3, 'ub', color='white',
                     bbox=dict(facecolor='blue', alpha=0.5))
        else:  # c == 92
            plt.plot(x, y, 'g.', markersize=10)
            plt.text(x+3, y+3, 'lb', color='white',
                     bbox=dict(facecolor='green', alpha=0.5))

    plt.axis('off')
    return plt.gcf()

def build_court_points_from_config(config, perspective):
    """
    Reproduce the logic from your generate_binary_heatmaps snippet
    for ub, lb:
        ub -> (border + int(5.25*10), border + court_width//2)
        lb -> (border + court_length - int(5.25*10), border + court_width//2)
    plus the 91 grid points from generate_perspective_aware_grid_points.
    """
    data_cfg = config.get('data', {})
    court_width = data_cfg.get('court_width', 500)
    court_length = data_cfg.get('court_length', 940)
    border = data_cfg.get('border', 0)
    grid_size = tuple(data_cfg.get('grid_size', [7,13]))

    if perspective:
        # 1) Generate the 91 grid points
        grid_points = generate_perspective_aware_grid_points(
            court_width,
            court_length,
            border,
            grid_size
        )  # returns a list of 91 (x, y)

        # 2) UB / LB as from your snippet
        ub_coord = (border + int(5.25*10), border + court_width // 2)
        lb_coord = (border + court_length - int(5.25*10), border + court_width // 2)

        # Combine => (93,2)
        all_points = np.array(list(grid_points) + [ub_coord, lb_coord], dtype=np.float32)
    else:
        grid_points = calculate_court_points(
            court_width,
            court_length,
            border
        )[0]
        points = []
        for grid_point in grid_points:
            points.append(grid_points[grid_point])
        all_points = np.array(list(points), dtype=np.float32)
    return all_points

def load_ground_truth(image_path, data_prep):
    """
    If you still want to visualize ground truth, adapt from your script
    to read an .npz, then resize to data_prep.output_size.
    Return shape (93, H_out, W_out).
    """
    from pathlib import Path
    label_path = Path(str(image_path).replace("/images/", "/labels/").replace(".jpg", ".npz"))
    if not label_path.exists():
        print(f"[INFO] No ground truth found at: {label_path}")
        return None
    
    npz_data = np.load(label_path, allow_pickle=True)
    gt = np.zeros((93, data_prep.output_size[1], data_prep.output_size[0]), dtype=np.float32)

    # Fill in grid_0..grid_90, ub, lb, etc. just like your code if you want
    # ...
    return gt  # or None if not found

def visualize_court_points(court_points, court_width, court_length, output_path):
    """
    Create a visualization of the court points on a blank canvas.
    
    Args:
        court_points: numpy array of shape (93, 2) containing (x, y) coordinates
        court_width: width of the court in pixels
        court_length: length of the court in pixels
        output_path: path to save the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure with white background
    plt.figure(figsize=(15, 8))
    plt.gca().set_facecolor('white')
    
    # Plot grid points (first 91 points)
    grid_points = court_points[:91]
    plt.scatter(grid_points[:, 0], grid_points[:, 1], 
               color='red', s=30, label='Grid Points')
    
    # Add small indices for grid points
    for i, (x, y) in enumerate(grid_points):
        plt.annotate(f'{i}', (x, y), xytext=(3, 3), 
                    textcoords='offset points', fontsize=6)
    
    # Plot basket points (last 2 points)
    basket_points = court_points[91:]
    plt.scatter(basket_points[:, 0], basket_points[:, 1], 
               color=['blue', 'green'], s=100, 
               label=['Upper Basket', 'Lower Basket'])
    
    # Add labels for baskets
    labels = ['UB', 'LB']
    for (x, y), label in zip(basket_points, labels):
        plt.annotate(label, (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=10)
    
    # Set bounds slightly larger than court dimensions
    margin = 50
    plt.xlim(-margin, court_length + margin)
    plt.ylim(-margin, court_width + margin)
    
    plt.title('Court Points Visualization')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved court points visualization: {output_path}")

def visualize_court_points(court_points, court_width, court_length, output_path):
    """
    Create a visualization of court points on a blank canvas.
    
    Args:
        court_points: numpy array of shape (N, 2) containing (x, y) coordinates
        court_width: width of the court in pixels
        court_length: length of the court in pixels
        output_path: path to save the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure with white background
    plt.figure(figsize=(15, 8))
    plt.gca().set_facecolor('white')
    
    # Plot all points
    plt.scatter(court_points[:, 0], court_points[:, 1], 
               color='blue', s=30, alpha=0.6)
    
    # Add indices for all points
    for i, (x, y) in enumerate(court_points):
        plt.annotate(f'{i}', (x, y), xytext=(3, 3), 
                    textcoords='offset points', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    # Set bounds slightly larger than court dimensions
    margin = 50
    plt.xlim(-margin, court_length + margin)
    plt.ylim(-margin, court_width + margin)
    
    # Add title
    plt.title('Court Points Visualization')
    
    # Add grid for better visualization
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add court dimensions
    plt.xlabel(f'Length: {court_length} pixels')
    plt.ylabel(f'Width: {court_width} pixels')
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved court points visualization to: {output_path}")

def get_midpoint(points, idx1, idx2):
    """Get the midpoint between two points."""
    p1 = points[idx1]
    p2 = points[idx2]
    return (p1 + p2) / 2

def visualize_all_points_overlay(image, predicted_points, predictions_tensor, output_path):
    """
    Create a visualization of all 93 predicted points overlaid on the input image,
    including confidence values for each point.
    
    Args:
        image: RGB image array
        predicted_points: numpy array of shape (93, 2) containing (x, y) coordinates
        predictions_tensor: Model output tensor of shape (93, H, W) containing heatmaps
        output_path: path to save the visualization
    """
    # Create a copy of the image for drawing
    draw_img = image.copy()
    
    # Convert predictions tensor to numpy if needed
    if hasattr(predictions_tensor, 'cpu'):
        predictions_tensor = predictions_tensor.cpu().numpy()
    
    # Get confidence values (maximum value from each heatmap)
    confidences = predictions_tensor.max(axis=(1, 2))
    
    # Draw grid points (first 91 points)
    for i in range(91):
        x, y = predicted_points[i].astype(int)
        conf = confidences[i]
        
        # Color based on confidence (red->green)
        color = (int(255 * (1-conf)), int(255 * conf), 0)  # BGR format
        
        # Draw point
        cv2.circle(draw_img, (x, y), 3, color, -1)
        
        # Add index label and confidence
        label = f"{i} ({conf:.2f})"
        cv2.putText(draw_img, label, (x + 3, y + 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Draw basket points (last 2 points) with different colors
    # Upper basket (91)
    x, y = predicted_points[91].astype(int)
    conf = confidences[91]
    cv2.circle(draw_img, (x, y), 5, (255, 0, 0), -1)  # Blue dot
    label = f"UB ({conf:.2f})"
    cv2.putText(draw_img, label, (x + 3, y + 3),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Lower basket (92)
    x, y = predicted_points[92].astype(int)
    conf = confidences[92]
    cv2.circle(draw_img, (x, y), 5, (0, 255, 0), -1)  # Green dot
    label = f"LB ({conf:.2f})"
    cv2.putText(draw_img, label, (x + 3, y + 3),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add a legend with confidence color scale
    height = draw_img.shape[0]
    for i in range(100):
        conf = i / 100
        color = (int(255 * (1-conf)), int(255 * conf), 0)
        pos = (10, height - 20 - i)
        cv2.line(draw_img, pos, (30, height - 20 - i), color, 1)
    
    # Add confidence scale labels
    cv2.putText(draw_img, "1.0", (35, height - 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(draw_img, "0.0", (35, height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(draw_img, "Conf:", (35, height - 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Save the visualization
    cv2.imwrite(str(output_path), cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved points overlay visualization with confidences: {output_path}")

def visualize_court_points_with_target(court_points, court_width, court_length, output_path, target_point=None):
    """
    Create a visualization of the court points with a highlighted target point.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(15, 8))
    plt.gca().set_facecolor('white')
    
    # Plot grid points (first 91 points)
    grid_points = court_points[:91]
    plt.scatter(grid_points[:, 0], grid_points[:, 1], 
               color='red', s=30, label='Grid Points')
    
    # Add small indices for grid points
    for i, (x, y) in enumerate(grid_points):
        plt.annotate(f'{i}', (x, y), xytext=(3, 3), 
                    textcoords='offset points', fontsize=6)
    
    # Plot basket points (last 2 points)
    basket_points = court_points[91:]
    plt.scatter(basket_points[:, 0], basket_points[:, 1], 
               color=['blue', 'green'], s=100, 
               label=['Upper Basket', 'Lower Basket'])
    
    # Add labels for baskets
    labels = ['UB', 'LB']
    for (x, y), label in zip(basket_points, labels):
        plt.annotate(label, (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=10)
    
    if target_point is not None:
        # Plot target point with a larger red circle
        plt.scatter([target_point[0]], [target_point[1]], 
                    color='red', s=200, marker='o', 
                    label='Target Point', zorder=5)
    
    # Set bounds slightly larger than court dimensions
    margin = 50
    plt.xlim(-margin, court_length + margin)
    plt.ylim(-margin, court_width + margin)
    
    plt.title('Court Points Visualization with Target Point')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved court points visualization with target: {output_path}")

def project_point_to_image(point, H_court_to_img):
    """Project a point from court coordinates to image coordinates using homography."""
    pt_court = np.array([point[0], point[1], 1], dtype=np.float32)
    pt_img = H_court_to_img @ pt_court
    pt_img /= pt_img[2]
    return (int(round(pt_img[0])), int(round(pt_img[1])))

def project_points_to_image(points_dict, H_court_to_img, img_shape):
    """Project a dictionary of points from court coordinates to image coordinates using homography."""
    projected_points = {}
    for key, point in points_dict.items():
        pt_court = np.array([point[0], point[1], 1], dtype=np.float32)
        pt_img = H_court_to_img @ pt_court
        pt_img /= pt_img[2]
        
        # Ensure point is within image bounds
        x = int(round(pt_img[0]))
        y = int(round(pt_img[1]))
        if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
            projected_points[key] = (x, y)
    
    return projected_points

def visualize_projected_points(image, points_dict, output_path, radius=5):
    """
    Draw all projected points on the image with labels.
    
    Args:
        image: RGB image array
        points_dict: Dictionary of point names to (x,y) coordinates
        output_path: Path to save the visualization
        radius: Radius of the circles to draw (default 5 pixels)
    """
    # Convert RGB to BGR for OpenCV
    draw_img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    
    # Define some colors to cycle through (in BGR)
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0)   # Cyan
    ]
    
    # Sort points by key for consistent coloring
    for i, (key, point) in enumerate(sorted(points_dict.items())):
        # Cycle through colors
        color = colors[i % len(colors)]
        
        # Draw circle
        cv2.circle(draw_img, point, radius, color, -1)
        
        # Add label with small connecting line
        label_offset = (point[0] + radius + 2, point[1] + radius + 2)
        cv2.line(draw_img, point, label_offset, color, 1)
        cv2.putText(draw_img, key, label_offset, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Save the visualization
    cv2.imwrite(str(output_path), draw_img)
    print(f"[INFO] Saved multi-point projection visualization: {output_path}")

def run_inference(model_path, image_path, output_dir, config_path=None):
    """
    Complete inference function that generates multiple visualizations including
    all court points projected onto the image.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load config
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Create data prep
    data_prep = KaliCalibDataPrep(config)

    # 2) Load & preprocess
    image_tensor, processed_image = load_and_preprocess_image(image_path, data_prep)
    h_img, w_img = processed_image.shape[:2]

    # 3) Device + model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)

    model = KaliCalibNet(n_keypoints=93).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # 4) Visualize predicted keypoints heatmap
    fig_pred = visualize_keypoints(processed_image, predictions, stride=data_prep.output_stride)
    pred_out = output_dir / f"{Path(image_path).stem}_pred.png"
    fig_pred.savefig(str(pred_out), bbox_inches='tight', pad_inches=0)
    plt.close(fig_pred)
    print(f"[INFO] Saved keypoints heatmap visualization: {pred_out}")

    # 5) Build court points
    perspective_court_points = build_court_points_from_config(config, True)
    
    # Get all target points from court points calculation
    data_cfg = config.get('data', {})
    court_width = data_cfg.get('court_width', 500)
    court_length = data_cfg.get('court_length', 940)
    border = data_cfg.get('border', 0)
    target_points_dict = calculate_court_points(
        court_width,
        court_length,
        border
    )[0]

    # 6) Extract predicted points and create points overlay visualization
    predicted_points = extract_predicted_points(
        predictions, 
        stride=data_prep.output_stride,
        num_keypoints=93
    )
    
    points_overlay_path = output_dir / f"{Path(image_path).stem}_points_overlay.jpg"
    visualize_all_points_overlay(processed_image, predicted_points, predictions, points_overlay_path)

    # 7) Compute homography and project all target points
    if predicted_points.shape != perspective_court_points.shape:
        print(f"[WARNING] Shape mismatch: predicted={predicted_points.shape}, court={perspective_court_points.shape}")
    else:
        H_img_to_court, inliers = cv2.findHomography(
            predicted_points, perspective_court_points, cv2.RANSAC, 5.0
        )
        
        if H_img_to_court is None:
            print("[WARNING] Homography computation failed.")
        else:
            # Get inverse homography for projecting court points to image
            H_court_to_img = np.linalg.inv(H_img_to_court)
            
            # Project all target points to image
            projected_points = project_points_to_image(
                target_points_dict, 
                H_court_to_img, 
                processed_image.shape
            )
            
            # Create visualization with all projected points
            target_img_out = output_dir / f"{Path(image_path).stem}_target_projected.jpg"
            visualize_projected_points(
                processed_image,
                projected_points,
                target_img_out,
                radius=5
            )
    
    print(f"\n[INFO] All visualizations have been saved to: {output_dir}")
    return predicted_points, perspective_court_points  # Return points for potential further analysis

###############################################################################
# UTILITY
###############################################################################

def load_and_preprocess_image(image_path, data_prep):
    """
    Loads an image, resizes to (data_prep.input_size), applies data_prep.transform => (1,C,H,W).
    Returns (tensor, rgb_image).
    """
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # If the image doesn't match the config size, do a resize
    if rgb_img.shape[:2] != (data_prep.input_size[1], data_prep.input_size[0]):
        rgb_img = cv2.resize(rgb_img, data_prep.input_size)

    # Keep copy for drawing
    processed_img = rgb_img.copy()

    # Convert to tensor
    img_tensor = data_prep.transform(rgb_img)  # shape (C,H,W)
    img_tensor = img_tensor.unsqueeze(0)       # (1,C,H,W)
    return img_tensor, processed_img

def main():
    parser = argparse.ArgumentParser(description="Inference script that uses perspective-aware grid + ub/lb from config.")
    parser.add_argument("--model", required=True, help="Path to trained .pth model file")
    parser.add_argument("--image", required=True, help="Path to input image file")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs")
    parser.add_argument("--config", default=None, help="Optional path to config .yaml")

    args = parser.parse_args()
    run_inference(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.output_dir,
        config_path=args.config
    )

if __name__ == "__main__":
    main()
