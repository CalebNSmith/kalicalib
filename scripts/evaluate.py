import argparse
from pathlib import Path
import yaml
import torch
import cv2
import numpy as np
from tqdm import tqdm
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.network import KaliCalibNet
from src.utils.visualization import visualize_keypoints

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def calculate_mse(pred_points, gt_points):
    """Calculate Mean Squared Error between predicted and ground truth points."""
    pred_points = np.array(pred_points)
    gt_points = np.array(gt_points)
    return np.mean(np.sum((pred_points - gt_points) ** 2, axis=1))

def evaluate_image(image_path, label_path, model, config, output_dir=None):
    """Evaluate model on a single image."""
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Load ground truth heatmaps from .npz
    try:
        with np.load(str(label_path)) as data:
            # Initialize heatmaps array
            n_keypoints = 93  # grid_0 through grid_90
            heatmap_h = img.shape[0] // 4  # 1/4 resolution
            heatmap_w = img.shape[1] // 4
            gt_heatmaps = np.zeros((n_keypoints + 1, heatmap_h, heatmap_w), dtype=np.float32)
            
            # Load and resize background channel
            if 'background' in data:
                bg = data['background']
                bg_resized = cv2.resize(bg, (heatmap_w, heatmap_h))
                gt_heatmaps[0] = bg_resized
            
            # Load and resize grid point channels
            for i in range(n_keypoints):
                key = f'grid_{i}'
                if key in data:
                    grid = data[key]
                    grid_resized = cv2.resize(grid, (heatmap_w, heatmap_h))
                    gt_heatmaps[i + 1] = grid_resized
    except Exception as e:
        raise KeyError(f"Error loading heatmaps from {label_path}: {str(e)}")

    # Extract ground truth keypoints from heatmaps
    gt_points = []
    for i in range(1, gt_heatmaps.shape[0]):  # Skip background channel
        heatmap = gt_heatmaps[i]
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        gt_points.append([x * 4, y * 4])  # Scale back to original resolution

    # Get model prediction
    with torch.no_grad():
        # Preprocess image
        resized = cv2.resize(img, tuple(config['model']['input_size']))
        input_tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0)
        input_tensor = input_tensor.to(next(model.parameters()).device)

        # Get prediction
        pred_heatmaps = model(input_tensor)

    # Convert predicted heatmaps to point coordinates
    pred_points = []
    for i in range(1, pred_heatmaps.size(1)):  # Skip background channel
        heatmap = pred_heatmaps[0, i].cpu().numpy()
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        pred_points.append([x * 4, y * 4])  # Scale back to original resolution

    print("Image:", Path(image_path).stem)
    for i in range(min(len(pred_points), len(gt_points))):
        print(f"  Predicted: {pred_points[i]}, Ground Truth: {gt_points[i]}")

    # Calculate metrics
    mse = calculate_mse(pred_points, gt_points)

    # Visualize if output directory is provided
    # if output_dir:
    #     vis_img = visualize_keypoints(img, pred_points, color=(0, 255, 0))  # Green for predictions
    #     vis_img = visualize_keypoints(vis_img, gt_points,
    #                                 color=(0, 0, 255))  # Red for ground truth

    #     output_path = Path(output_dir) / f"{Path(image_path).stem}_eval.jpg"
    #     cv2.imwrite(str(output_path), vis_img)

    return mse

def main():
    parser = argparse.ArgumentParser(description='Evaluate KaliCalib model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output-dir', type=str,
                       help='Path to output directory for visualizations')
    args = parser.parse_args()

    setup_logging()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KaliCalibNet(config['model']['n_keypoints']).to(device)
    
    # Load checkpoint and handle both direct state dict and wrapped formats
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint contains wrapped state dict
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Checkpoint contains direct state dict
        model.load_state_dict(checkpoint)
    
    model.eval()

    # Evaluate all images
    data_dir = Path(args.data_dir)
    mse_values = []

    for label_path in tqdm(list((data_dir / 'labels').glob('*.npz'))):
        image_path = data_dir / 'images' / f"{label_path.stem}.jpg"
        try:
            mse = evaluate_image(image_path, label_path, model, config, args.output_dir)
            print(mse)
            exit()
            mse_values.append(mse)
        except FileNotFoundError:
            logging.error(f"Label file not found: {label_path}")
        except KeyError as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error processing {image_path}: {str(e)}")
            exit()

    # Report results
    if mse_values:
        avg_mse = np.mean(mse_values)
        logging.info(f"Evaluation complete:")
        logging.info(f"Average MSE: {avg_mse:.2f}")
        logging.info(f"Number of images processed: {len(mse_values)}")
    else:
        logging.warning("No images were successfully processed for evaluation.")

if __name__ == '__main__':
    main()