import cv2
import yaml
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Import your data preparation class
from src.data.data_prep import KaliCalibDataPrep

# Import your utility functions:
from src.utils.court import (
    calculate_court_points,
    draw_court_lines,
    transform_point,
    prepare_point_pairs,
    draw_keypoints
)

def load_config(config_path):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_image_and_label(image_path, label_path):
    """Load image and corresponding label file."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    
    return image, label_data

def estimate_homography(label_points, court_width, court_length, border):
    """Estimate homography using all available labeled points."""
    # Get the reference court points
    court_points, _, _ = calculate_court_points(court_width, court_length, border)
    
    src_points = []  # Court space points
    dst_points = []  # Image space points
    
    # Image dimensions (assumed 1080p for this example)
    h, w = 1080, 1920
    
    # For each labeled point, if it exists in our court points, add the pair
    for point_name, point_data in label_points['points'].items():
        if point_name in court_points:
            # Convert normalized coords to pixel coords
            image_point = (int(point_data['x'] * w), int(point_data['y'] * h))
            court_point = court_points[point_name]
            
            src_points.append(court_point)
            dst_points.append(image_point)
    
    if len(src_points) < 4:
        raise ValueError(
            f"Need at least 4 points to estimate homography. Only found {len(src_points)}"
        )
    
    print(f"Using {len(src_points)} points for homography estimation")
    
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    # Court space -> image space homography
    H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    if status is not None:
        inliers = np.sum(status)
        print(f"RANSAC found {inliers} inliers out of {len(status)} points")
    
    return H

def main():
    # 1. Load configuration
    config = load_config('configs/default.yaml')
    data_prep = KaliCalibDataPrep(config)
    
    # 2. Output directory
    output_dir = Path('debug_output')
    output_dir.mkdir(exist_ok=True)
    
    # 3. Load sample image/label
    image_path = Path('/home/leb/basketball/data/charleston_vs_west_liberty-12-11-2024/images/10_150.jpg')
    label_path = Path('/home/leb/basketball/data/charleston_vs_west_liberty-12-11-2024/labels/10_150.json')
    image, label_data = load_image_and_label(image_path, label_path)
    
    # 4. Debug grid
    court_grid = data_prep.debug_grid()
    plt.imsave(output_dir / 'court_grid.png', court_grid, cmap='gray')
    
    # 5. Estimate homography
    H = estimate_homography(
        label_data,
        config['data']['court_width'],
        config['data']['court_length'],
        config['data']['border']
    )
    print("Estimated homography matrix:")
    print(H)
    
    # 6. (Optional) Image with labeled points
    h_img, w_img = image.shape[:2]
    labeled_points_img = image.copy()
    for point_name, point_data in label_data['points'].items():
        px = int(point_data['x'] * w_img)
        py = int(point_data['y'] * h_img)
        cv2.circle(labeled_points_img, (px, py), 5, (0, 0, 255), -1)
    labeled_points_path = output_dir / 'original_image_with_labeled_points.png'
    cv2.imwrite(str(labeled_points_path), labeled_points_img)
    
    # 7. Draw court lines directly on the original image (no warping)
    #    We transform each court point from "court space" -> "image space" using H
    court_points, circle_radius, three_point_radius = calculate_court_points(
        config['data']['court_width'],
        config['data']['court_length'],
        config['data']['border']
    )
    lines_overlay_img = image.copy()  # so we don't modify the original in memory
    draw_court_lines(lines_overlay_img, H, court_points, circle_radius, three_point_radius)

    # Save the original resolution image with lines
    overlay_path = output_dir / 'original_image_with_court_lines.png'
    cv2.imwrite(str(overlay_path), lines_overlay_img)
    print(f"Saved lines overlay to {overlay_path}")
    
    # 8. Visualize transformed points
    transformed_viz = data_prep.visualize_points(
        image,
        H,
        save_path=str(output_dir / 'transformed_points.png')
    )
    
    # 9. Generate & visualize heatmaps
    normalized_image, heatmaps = data_prep.generate_heatmaps(image, H)
    for i in range(min(5, heatmaps.shape[0])):
        heatmap_path = output_dir / f'heatmap_{i}.png'
        plt.imsave(heatmap_path, heatmaps[i].numpy(), cmap='hot')
    
    print(f"Debug visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
