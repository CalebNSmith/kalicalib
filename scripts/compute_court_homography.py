import numpy as np
import cv2
import json
from pathlib import Path

def calculate_court_points(image_width, image_height, border):
    """
    Calculate standard court points in image coordinate system.
    
    Args:
        image_width (int): Width of the target image
        image_height (int): Height of the target image
        border (int): Border size in pixels
        
    Returns:
        Dictionary of point names to (x, y) coordinates
    """
    # Calculate dimensions based on image size
    court_width = image_height - 2 * border
    court_length = image_width - 2 * border
    
    # Convert measurements to proportion of court width/length then to pixels
    def feet_to_pixels(feet, reference_size):
        return int((feet / 94) * reference_size)  # 94 feet is regulation court length
        
    corner_3pt_from_baseline = feet_to_pixels(9 + 10/12 + 3/8/12, court_length)
    corner_3pt_from_sideline = feet_to_pixels(40.125/12, court_width)
    key_width = feet_to_pixels(12, court_width)
    key_height = feet_to_pixels(19, court_length)
    circle_radius = feet_to_pixels(6, min(court_width, court_length))
    three_point_radius = feet_to_pixels(22 + 1/12 + 3/4/12, court_length)
    basket_offset = feet_to_pixels(5 + 3/12, court_length)
    baseline_mark_distance = feet_to_pixels(19, court_width)
    sideline_mark_distance = feet_to_pixels(28, court_length)
    sideline_mark_inset = feet_to_pixels(3, court_width)
    peak_distance = feet_to_pixels(27 + 4/12 + 3/4/12, court_length)  # 5'3" + 22'1¾"
    
    # Center court coordinates
    center_x = border + court_length//2
    center_y = border + court_width//2
    
    # Initialize points dictionary with same structure as original
    points = {
        # Upper baseline points
        'ubl': (border, border),
        'ubr': (border, border + court_width),
        
        # Upper 3-point line points
        'u3l': (border + corner_3pt_from_baseline, border + corner_3pt_from_sideline),
        'u3r': (border + corner_3pt_from_baseline, border + court_width - corner_3pt_from_sideline),
        'u3p': (border + peak_distance, border + court_width//2),
        'u3bl': (border, border + corner_3pt_from_sideline),
        'u3br': (border, border + court_width - corner_3pt_from_sideline),
        
        # Upper key points
        'ukl': (border + key_height, border + (court_width - key_width)//2),
        'ukr': (border + key_height, border + (court_width + key_width)//2),
        
        # Center court points
        'cc': (center_x, center_y),
        'ml': (center_x, border),
        'mr': (center_x, border + court_width),
        
        # Lower baseline points
        'lbl': (border + court_length, border),
        'lbr': (border + court_length, border + court_width),
        
        # Lower 3-point line points
        'l3l': (border + court_length - corner_3pt_from_baseline, border + corner_3pt_from_sideline),
        'l3r': (border + court_length - corner_3pt_from_baseline, border + court_width - corner_3pt_from_sideline),
        'l3p': (border + court_length - peak_distance, border + court_width//2),
        'l3bl': (border + court_length, border + corner_3pt_from_sideline),
        'l3br': (border + court_length, border + court_width - corner_3pt_from_sideline),
        
        # Lower key points
        'lkl': (border + court_length - key_height, border + (court_width - key_width)//2),
        'lkr': (border + court_length - key_height, border + (court_width + key_width)//2),
        
        # Basket points
        'ub': (border + basket_offset, border + court_width//2),
        'lb': (border + court_length - basket_offset, border + court_width//2),
        
        # Baseline marks
        'ubml': (border, border + baseline_mark_distance),
        'ubmr': (border, border + court_width - baseline_mark_distance),
        'lbml': (border + court_length, border + baseline_mark_distance),
        'lbmr': (border + court_length, border + court_width - baseline_mark_distance),
        
        # Center circle points
        'cct': (center_x, center_y - circle_radius),
        'ccb': (center_x, center_y + circle_radius),
        'ccl': (center_x - circle_radius, center_y),
        'ccr': (center_x + circle_radius, center_y),
        
        # Sideline marks
        'usml': (border + sideline_mark_distance, border + sideline_mark_inset),
        'usmr': (border + court_length - sideline_mark_distance, border + sideline_mark_inset),
        'usml_side': (border + sideline_mark_distance, border),
        'usmr_side': (border + court_length - sideline_mark_distance, border),
    }
    
    return points, circle_radius, three_point_radius

def compute_grid_homography(image_size=(960, 540), court_size=(940, 500), border=30, grid_size=(10, 20)):
    """
    Compute homography from image coordinates to court coordinates.
    
    Args:
        image_size (tuple): Target image size (width, height)
        court_size (tuple): Court size in tenths of feet (length, width)
        border (int): Border size in tenths of feet
        grid_size (tuple): Number of rows and columns in the grid
    
    Returns:
        numpy.ndarray: Homography matrix mapping from image to court coordinates
    """
    # Get court points in image coordinates
    image_points_dict, _, _ = calculate_court_points(image_size[0], image_size[1], border)
    
    # Select a subset of reliable points for homography computation
    key_points = ['ubl', 'ubr', 'lbl', 'lbr', 'cc', 'ub', 'lb']
    image_points = [image_points_dict[k] for k in key_points]
    
    # Create corresponding points in court coordinates (feet)
    court_length, court_width = court_size
    court_points = [
        (border, border),  # ubl
        (border, border + court_width),  # ubr
        (border + court_length, border),  # lbl
        (border + court_length, border + court_width),  # lbr
        (border + court_length//2, border + court_width//2),  # cc
        (border + int((5 + 3/12) * 10), border + court_width//2),  # ub
        (border + court_length - int((5 + 3/12) * 10), border + court_width//2),  # lb
    ]
    
    # Convert to numpy arrays
    src_points = np.float32(image_points)  # Image coordinates
    dst_points = np.float32(court_points)  # Court coordinates
    
    # Compute homography from image to court coordinates
    homography, _ = cv2.findHomography(src_points, dst_points)
    
    return homography

def save_grid_visualization(output_path, image_size=(960, 540), court_size=(940, 500), border=30):
    """
    Save visualization of the court points and compute homography.
    """
    # Get court points in image coordinates
    court_points_dict, circle_radius, three_point_radius = calculate_court_points(
        image_size[0], 
        image_size[1], 
        border
    )
    
    # Compute homography
    homography = compute_grid_homography(image_size, court_size, border)
    
    # Create blank image
    court_vis = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
    
    # Draw points
    for name, point in court_points_dict.items():
        x, y = point
        cv2.circle(court_vis, (int(x), int(y)), 3, (0, 0, 255), -1)
        # Add small text label
        cv2.putText(court_vis, name, (int(x)+5, int(y)+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    # Save visualization and data
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save visualization
    cv2.imwrite(str(output_path / 'court_grid.png'), court_vis)
    
    # Save homography matrix
    np.savez(output_path / 'homography.npz', H=homography)

def main():
    # Example usage
    output_path = Path('outputs/homography/')
    
    # Generate and save visualizations with default parameters
    save_grid_visualization(output_path)
    print(f"Grid visualization and homography saved to {output_path}")

if __name__ == "__main__":
    main()