import argparse
import json
import cv2
import numpy as np
from pathlib import Path
import yaml
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.court import calculate_court_points, transform_point, generate_perspective_aware_grid_points

def create_binary_mask(height, width, points, radius=10):
    """
    Create a binary mask from point locations.
    
    Args:
        height (int): Image height
        width (int): Image width
        points (np.ndarray): Array of (row, col) point coordinates
        radius (int): Radius of the circular mask around each point
        
    Returns:
        np.ndarray: Binary mask where 1s indicate foreground regions
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for row, col in points:
        cv2.circle(mask, (int(col), int(row)), radius, 1, -1)
    
    return mask

def generate_binary_heatmaps(image_path, label_path, config_path='configs/default.yaml', resize=False, 
                           output_dir=None, label_output=None, image_output=None):
    """
    Generates dense binary masks for keypoints and background using perspective-aware sampling.

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the JSON label file.
        config_path (str): Path to the configuration YAML file.
        resize (bool): Whether to resize the image to 960x540.
        output_dir (str, optional): Directory to save all outputs (NPZ and visualizations).
        label_output (str, optional): Path to save NPZ file.
        image_output (str, optional): Path to save processed image.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_prep_config = config['data']
    model_config = config['model']

    court_width = data_prep_config['court_width']
    court_length = data_prep_config['court_length']
    border = data_prep_config['border']
    grid_size = tuple(data_prep_config['grid_size'])

    # Load and potentially resize the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    
    original_height, original_width = image.shape[:2]
    
    if resize:
        target_width, target_height = 960, 540
        image = cv2.resize(image, (target_width, target_height))
        scale_x = target_width / original_width
        scale_y = target_height / original_height
    else:
        target_width, target_height = original_width, original_height
        scale_x = scale_y = 1.0

    with open(label_path, 'r') as f:
        label_data = json.load(f)

    # Calculate court points
    court_points_dict, _, _ = calculate_court_points(court_width, court_length, border)

    # Generate perspective-aware grid points in court coordinates
    grid_points_coords = generate_perspective_aware_grid_points(court_width, court_length, border, grid_size)
    print(f"Generated {len(grid_points_coords)} grid points")

    # Explicit keypoints in court coordinates
    explicit_keypoints_coords = {
        'ub': (border + int(5.25 * 10), border + court_width // 2),  # Upper basket
        'lb': (border + court_length - int(5.25 * 10), border + court_width // 2)  # Lower basket
    }

    # Get point correspondences for homography, adjusting for resize if necessary
    src_points = []
    dst_points = []
    for point_name, point_data in label_data['points'].items():
        if point_name in court_points_dict:
            # Scale the points if image was resized
            src_points.append([
                point_data['x'] * target_width,
                point_data['y'] * target_height
            ])
            dst_points.append(court_points_dict[point_name])

    if len(src_points) < 4:
        raise ValueError(f"Need at least 4 point correspondences to calculate homography, got {len(src_points)}")

    homography, _ = cv2.findHomography(np.float32(dst_points), np.float32(src_points))
    if homography is None:
        raise ValueError("Could not compute homography.")

    all_masks = {}
    
    # Process grid points
    for i, court_point in enumerate(grid_points_coords):
        transformed_point = transform_point(court_point, homography)
        locations = []
        if 0 <= transformed_point[1] < target_height and 0 <= transformed_point[0] < target_width:
            locations.append(transformed_point[::-1])  # Store as (row, col)
        mask = create_binary_mask(target_height, target_width, np.array(locations))
        all_masks[f'grid_{i}'] = mask

    # Process explicit keypoints
    for name, court_point in explicit_keypoints_coords.items():
        transformed_point = transform_point(court_point, homography)
        locations = []
        if 0 <= transformed_point[1] < target_height and 0 <= transformed_point[0] < target_width:
            locations.append(transformed_point[::-1])  # Store as (row, col)
        mask = create_binary_mask(target_height, target_width, np.array(locations))
        all_masks[name] = mask

    # Create background mask (inverse of all other masks combined)
    combined_mask = np.zeros((target_height, target_width), dtype=np.uint8)
    for mask in all_masks.values():
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    all_masks['background'] = 1 - combined_mask

    # Handle outputs based on provided arguments
    if output_dir:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save visualizations
        for name, mask in all_masks.items():
            viz = image.copy()
            viz[mask == 1] = [203, 192, 255]  # Pink tint for foreground
            viz_output_file = output_path / f"{Path(label_path).stem}_{name}_mask_visualization.jpg"
            cv2.imwrite(str(viz_output_file), viz)
            print(f"Mask visualization for {name} saved to: {viz_output_file}")

        # Save NPZ
        output_file = output_path / f"{Path(label_path).stem}_binary_masks.npz"
        np.savez_compressed(output_file, **all_masks)
        print(f"Binary masks saved to: {output_file}")
    
    else:
        # Save NPZ to specified label output path
        Path(label_output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(label_output, **all_masks)
        print(f"Binary masks saved to: {label_output}")
        
        # Save processed image to specified image output path
        Path(image_output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(image_output, image)
        print(f"Processed image saved to: {image_output}")


def main():
    parser = argparse.ArgumentParser(description="Generate binary masks from image and label.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--label", required=True, help="Path to the JSON label file.")
    parser.add_argument("--config", type=str, default='configs/default.yaml', help="Path to the configuration YAML file.")
    parser.add_argument("--resize", action="store_true", help="Resize image to 960x540")
    
    # Create mutually exclusive group for output options
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output-dir", help="Directory to save all outputs (NPZ and visualizations)")
    output_group.add_argument("--label-output", help="Path to save NPZ file", metavar="PATH")
    
    # Make image-output required if label-output is provided
    parser.add_argument("--image-output", help="Path to save processed image", metavar="PATH")

    args = parser.parse_args()
    
    # Validate arguments
    if args.label_output and not args.image_output:
        parser.error("--image-output is required when using --label-output")
    if args.image_output and not args.label_output:
        parser.error("--label-output is required when using --image-output")

    generate_binary_heatmaps(
        image_path=args.image,
        label_path=args.label,
        config_path=args.config,
        resize=args.resize,
        output_dir=args.output_dir,
        label_output=args.label_output,
        image_output=args.image_output
    )

if __name__ == "__main__":
    main()