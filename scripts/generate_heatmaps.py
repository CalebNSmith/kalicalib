import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np
import json
import logging
from tqdm import tqdm

from src.data.data_prep import KaliCalibDataPrep
from src.utils.visualization import visualize_heatmaps
from src.utils.court import calculate_court_points

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def process_image(label_path, data_prep, output_dir, visualize=True, overwrite=False):
    """Process a single image and generate heatmaps."""
    label_name = label_path.stem
    image_path = label_path.parent.parent / 'images' / f"{label_name}.jpg"
    
    # Updated paths for heatmaps and visualizations
    heatmap_path = output_dir / 'labels' / f"{label_name}_heatmaps.npz"
    
    if visualize:
        vis_path = output_dir / 'visualizations' / f"{label_name}_heatmap_vis.jpg"
        if vis_path.exists() and not overwrite:
            logging.info(f"Skipping {vis_path.name} - already exists")
            return False

    try:
        # Load and resize image - handle potential symlinks by copying the image
        try:
            if image_path.is_symlink():
                # Get the real path and read from there
                real_path = image_path.resolve()
                img = cv2.imread(str(real_path))
                if img is None:
                    raise ValueError(f"Could not load image from resolved path {real_path}")
                
                # Create a new image in the original location
                cv2.imwrite(str(image_path), img)
            else:
                img = cv2.imread(str(image_path))
                if img is None:
                    raise ValueError(f"Could not load image from {image_path}")
        except Exception as e:
            logging.error(f"Error processing {image_path.name}: {str(e)}")
            return False
        
        # Resize to 1024x1920
        img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        height, width = img.shape[:2]

        # Load label data
        with open(str(label_path), 'r') as f:
            data = json.load(f)
        
        # Calculate court points
        court_points, _, _ = calculate_court_points(
            data_prep.court_width,
            data_prep.court_length,
            data_prep.border
        )

        # Get point correspondences
        src_points, dst_points = [], []
        for point_name, point_data in data['points'].items():
            if point_name in court_points:
                src_points.append([point_data['x'] * width, point_data['y'] * height])
                dst_points.append(court_points[point_name])
                
        if len(src_points) < 4:
            raise ValueError(f"Need at least 4 point correspondences, got {len(src_points)}")

        homography, _ = cv2.findHomography(np.float32(dst_points), np.float32(src_points))
        if homography is None:
            raise ValueError("Could not calculate homography matrix")

        # Generate heatmaps
        heatmap_output = data_prep.generate_heatmaps(img, homography)
        
        # Depending on return type, extract heatmaps appropriately
        if isinstance(heatmap_output, tuple):
            norm_image, heatmaps = heatmap_output
        else:
            heatmaps = heatmap_output

        # Create directories if they don't exist
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save resized original image
        resized_image_path = output_dir / 'images' / f"{label_name}.jpg"
        cv2.imwrite(str(resized_image_path), img)
        
        # Save heatmaps
        np.savez_compressed(str(heatmap_path), heatmaps=heatmaps.numpy())

        # Create visualization if requested
        if visualize:
            vis = visualize_heatmaps(img, heatmaps.numpy())
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_path), vis)

        return True

    except Exception as e:
        logging.error(f"Error processing {image_path.name}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate heatmaps for training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--no-vis', action='store_true',
                       help='Disable visualization generation')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing files')
    args = parser.parse_args()

    setup_logging()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize data preparation
    data_prep = KaliCalibDataPrep(config)

    # Create output directory (now inside the game directory)
    output_dir = Path(args.data_dir) / 'heatmaps'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'labels').mkdir(exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    if not args.no_vis:
        (output_dir / 'visualizations').mkdir(exist_ok=True)

    # Process all images
    labels_dir = Path(args.data_dir) / 'labels'
    processed = 0
    total = 0

    for label_path in tqdm(list(labels_dir.glob('*.json'))):
        total += 1
        if process_image(label_path, data_prep, output_dir, not args.no_vis, args.overwrite):
            processed += 1

    logging.info(f"Processing complete: {processed}/{total} files processed")

if __name__ == '__main__':
    main()