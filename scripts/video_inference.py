import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Adjust these paths as needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.network import KaliCalibNet
from src.data.data_prep import KaliCalibDataPrep
from src.utils.court import calculate_court_points

# Import necessary functions from the image inference script
from inference import (
    build_court_points_from_config,
    extract_predicted_points,
    project_points_to_image
)

def draw_points_on_frame(frame, points_dict):
    """
    Draw points from points_dict on the frame with labels.
    
    Args:
        frame: BGR frame
        points_dict: Dictionary of points to draw
    
    Returns:
        frame: BGR frame with points drawn
    """
    vis_frame = frame.copy()
    
    # Draw each point
    for label, point in points_dict.items():
        x, y = point
        # Draw circle
        cv2.circle(vis_frame, (x, y), 3, (0, 255, 0), -1)
        # Add label
        cv2.putText(vis_frame, str(label), (x + 5, y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                   
    return vis_frame

def calculate_court_distance(point1, point2, H_img_to_court):
    """
    Calculate the distance between two points in court space.
    
    Args:
        point1: First point in image coordinates [x, y]
        point2: Second point in image coordinates [x, y]
        H_img_to_court: Homography matrix from image to court space
        
    Returns:
        float: Distance in court space units
    """
    # Convert points to homogeneous coordinates
    p1_homog = np.array([[point1[0]], [point1[1]], [1]])
    p2_homog = np.array([[point2[0]], [point2[1]], [1]])
    
    # Transform points to court space
    p1_court = H_img_to_court @ p1_homog
    p2_court = H_img_to_court @ p2_homog
    
    # Convert from homogeneous coordinates
    p1_court = (p1_court / p1_court[2])[:2].flatten()
    p2_court = (p2_court / p2_court[2])[:2].flatten()
    
    # Calculate Euclidean distance
    return float(np.sqrt(np.sum((p1_court - p2_court) ** 2)))

def process_frame(frame, model, data_prep, court_points, target_points_dict):
    """
    Process a single frame and return the projected points and court distance.
    
    Args:
        frame: RGB frame
        model: loaded KaliCalibNet model
        data_prep: KaliCalibDataPrep instance
        court_points: numpy array of perspective-aware court points
        target_points_dict: dictionary of target points to project
        
    Returns:
        tuple: (projected_points dict, court_distance float) or (None, None) if homography fails
    """
    # Resize frame if needed
    if frame.shape[:2] != (data_prep.input_size[1], data_prep.input_size[0]):
        frame = cv2.resize(frame, data_prep.input_size)
    
    # Preprocess frame
    processed_frame = frame.copy()
    frame_tensor = data_prep.transform(frame)
    frame_tensor = frame_tensor.unsqueeze(0)
    
    # Move to device
    device = next(model.parameters()).device
    frame_tensor = frame_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(frame_tensor)[0]
    
    # Extract predicted points
    predicted_points = extract_predicted_points(
        predictions,
        stride=data_prep.output_stride,
        num_keypoints=93
    )
    
    # Compute homography
    H_img_to_court, inliers = cv2.findHomography(
        predicted_points, court_points, cv2.RANSAC, 5.0
    )
    
    if H_img_to_court is None:
        return None, None
        
    # Calculate distance between specified court points
    point1 = np.array([235, 250])
    point2 = np.array([705, 250])
    court_distance = calculate_court_distance(point1, point2, H_img_to_court)
    
    # Get inverse homography
    H_court_to_img = np.linalg.inv(H_img_to_court)
    
    # Project all target points to image
    projected_points = project_points_to_image(
        target_points_dict,
        H_court_to_img,
        processed_frame.shape
    )
    
    return projected_points, court_distance

def run_video_inference(model_path, video_path, output_dir, debug=False, config_path=None):
    """
    Run inference on every frame of a video and save projected points to JSON.
    
    Args:
        model_path: Path to trained .pth model
        video_path: Path to input video file
        output_dir: Directory to save output files
        debug: If True, save visualizations of each frame
        config_path: Optional path to config YAML
    """
    # Load config
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Setup data prep and model
    data_prep = KaliCalibDataPrep(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = KaliCalibNet(n_keypoints=93).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Get court points
    court_points = build_court_points_from_config(config, True)
    
    # Get target points dictionary
    data_cfg = config.get('data', {})
    court_width = data_cfg.get('court_width', 500)
    court_length = data_cfg.get('court_length', 940)
    border = data_cfg.get('border', 0)
    target_points_dict = calculate_court_points(
        court_width,
        court_length,
        border
    )[0]
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract video filename without extension
    video_basename = Path(video_path).stem
    
    # Dictionary to store results
    results = {
        'video_info': {
            'path': str(video_path),
            'total_frames': total_frames,
            'fps': fps,
            'distances': []  # List to store court distances
        },
        'frames': {}
    }
    
    # Process each frame
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        frame_idx = 0
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB for model input
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            projected_points, court_distance = process_frame(
                rgb_frame,
                model,
                data_prep,
                court_points,
                target_points_dict
            )
            
            # Store results
            if projected_points is not None:
                # Convert points to list format for JSON serialization
                frame_points = {
                    k: list(map(int, v)) for k, v in projected_points.items()
                }
                results['frames'][str(frame_idx)] = frame_points
                results['video_info']['distances'].append(court_distance)
                
                # Save debug frame if enabled
                if debug:
                    debug_frame = draw_points_on_frame(bgr_frame, frame_points)
                    debug_path = output_dir / f"{video_basename}_{frame_idx:04d}.jpg"
                    cv2.imwrite(str(debug_path), debug_frame)
            else:
                # If homography failed, append None for this frame
                results['video_info']['distances'].append(None)
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    
    # Save results to JSON
    json_path = output_dir / f"{video_basename}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[INFO] Results saved to: {json_path}")
    print(f"Successfully processed frames: {len(results['frames'])}/{total_frames}")
    if debug:
        print(f"Debug frames saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run court point detection on video frames")
    parser.add_argument("--model", required=True, help="Path to trained .pth model file")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output-dir", required=True, help="Directory to save output files")
    parser.add_argument("--debug", action="store_true", help="Save debug frames with point overlays")
    parser.add_argument("--config", default=None, help="Optional path to config .yaml")
    
    args = parser.parse_args()
    run_video_inference(
        model_path=args.model,
        video_path=args.video,
        output_dir=args.output_dir,
        debug=args.debug,
        config_path=args.config
    )

if __name__ == "__main__":
    main()