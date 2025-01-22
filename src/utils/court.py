import math
import cv2
import numpy as np

def prepare_point_pairs(label_data, image_width, image_height, court_points):
    """
    For each labeled point in `label_data`, if the point name is in `court_points`,
    pair up the (x,y) in image pixels with the court coordinate system (court_points).
    Returns two arrays:
      - src_points: the points in the image (pixel) space
      - dst_points: the corresponding points in the 'court' space
    """
    src_pts = []
    dst_pts = []

    # label_data might be something like: 
    # {
    #   "points": {
    #       "ubl": {"x": 0.05, "y": 0.03},
    #       "ubr": {"x": 0.95, "y": 0.03},
    #       ...
    #   }
    # }
    for point_name, point_data in label_data["points"].items():
        if point_name in court_points:
            # Convert normalized coords (0..1) to pixel coords
            px = int(point_data["x"] * image_width)
            py = int(point_data["y"] * image_height)

            # Court-space point
            cx, cy = court_points[point_name]

            src_pts.append([px, py])
            dst_pts.append([cx, cy])

    # Convert lists to float32 NumPy arrays for cv2.findHomography
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    return src_pts, dst_pts


def draw_court_lines(image, homography, court_points, circle_radius, three_point_radius):
    """
    Draw standard basketball court lines onto 'image', based on the specified 'homography'.
    Uses 'transform_point' (already defined in this file) to reproject each point.
    """
    lines = [
        ('ubl', 'lbl'),
        ('lbl', 'lbr'),
        ('lbr', 'ubr'),
        ('ubr', 'ubl'),
        ('ml', 'mr'),
        ('ubml', 'ukl'),
        ('ubmr', 'ukr'),
        ('lbml', 'lkl'),
        ('lbmr', 'lkr'),
        ('ukl', 'ukr'),
        ('lkl', 'lkr'),
        ('u3bl', 'u3l'),
        ('u3br', 'u3r'),
        ('l3bl', 'l3l'),
        ('l3br', 'l3r'),
        ('ubml', 'ubl'),
        ('ubmr', 'ubr'),
        ('lbml', 'lbl'),
        ('lbmr', 'lbr')
    ]
    
    # 1. Straight lines
    for start_label, end_label in lines:
        start_point = court_points[start_label]
        end_point = court_points[end_label]
        start_transformed = transform_point(start_point, homography)
        end_transformed = transform_point(end_point, homography)
        cv2.line(image, start_transformed, end_transformed, (0, 0, 0), 5)
    
    # 2. 3-point arcs (upper and lower)
    for basket_label, start_label, end_label in [('ub', 'u3l', 'u3r'), ('lb', 'l3l', 'l3r')]:
        basket_center = court_points[basket_label]
        arc_points = []
        num_points = 30

        start_point = court_points[start_label]
        end_point = court_points[end_label]
        
        start_angle = math.atan2(start_point[1] - basket_center[1], start_point[0] - basket_center[0])
        end_angle = math.atan2(end_point[1] - basket_center[1], end_point[0] - basket_center[0])
        
        # Adjust angles so we move in a consistent direction
        if basket_label == 'ub':
            if end_angle < start_angle:
                end_angle += 2 * math.pi
        else:  # 'lb'
            if start_angle < end_angle:
                start_angle += 2 * math.pi

        for i in range(num_points + 1):
            t = i / num_points
            angle = start_angle * (1 - t) + end_angle * t
            x = basket_center[0] + three_point_radius * math.cos(angle)
            y = basket_center[1] + three_point_radius * math.sin(angle)
            point_transformed = transform_point((x, y), homography)
            arc_points.append(point_transformed)

        cv2.polylines(image, [np.array(arc_points)], False, (0, 0, 0), 5)
    
    # 3. Center circle
    center = court_points['cc']
    circle_points = []
    num_points = 36
    
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = center[0] + circle_radius * math.cos(angle)
        y = center[1] + circle_radius * math.sin(angle)
        point_transformed = transform_point((x, y), homography)
        circle_points.append(point_transformed)
    
    cv2.polylines(image, [np.array(circle_points)], True, (0, 0, 0), 5)


def draw_keypoints(image, homography, court_points):
    """
    Draw each labeled court keypoint onto 'image' based on the homography.
    """
    for point_name, point in court_points.items():
        transformed_point = transform_point(point, homography)
        cv2.circle(image, transformed_point, 5, (0, 0, 255), -1)
        cv2.putText(image, point_name,
                    (transformed_point[0] + 5, transformed_point[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def calculate_court_points(court_width, court_length, border):
    """
    Calculate standard court points in court coordinate system.
    All dimensions in tenths of feet.
    
    Returns:
        Dictionary of point names to (x, y) coordinates
    """
    # Convert measurements to tenths of feet (matches ground truth dimensions)
    corner_3pt_from_baseline = int((9 + 10/12 + 3/8/12) * 10)  # 9'10⅜" from baseline
    corner_3pt_from_sideline = int((40.125/12) * 10)  # 40⅛" from sideline
    key_width = 12 * 10  # 12' key width
    key_height = 19 * 10  # 19' key height
    circle_radius = 6 * 10  # 6' radius
    three_point_radius = int((22 + 1/12 + 3/4/12) * 10)  # 22'1¾"
    basket_offset = int((5 + 3/12) * 10)  # 5'3" from baseline
    baseline_mark_distance = 19 * 10  # 19' from corners
    sideline_mark_distance = 28 * 10  # 28' from baseline
    sideline_mark_inset = 3 * 10  # 3' in from sideline
    line_depth = int(8/12 * 10)  # 8 inches deep
    dist_from_baseline = int((3 + 1/12) * 10)  # 3'1" from baseline
    inward_dist = int(12/12 * 10)  # 12" inward

    # Additional marking distances
    additional_marks = [
        7 * 10,           # 7 feet
        8 * 10,           # 8 feet
        int((11 + 1/12) * 10), # 11'1"
        int((14 + 3/12) * 10), # 14'3"
        int((17 + 5/12) * 10)  # 17'5"
    ]

    # Calculate peak distance for 3-point line (5'3" + 22'1¾")
    peak_distance = int((5 + 3/12 + 22 + 1/12 + 3/4/12) * 10)
    
    # Center court coordinates for convenience
    center_x = border + court_length//2
    center_y = border + court_width//2
    
    # Initialize points dictionary with existing points
    points = {
        # [Previous points remain the same]
        'ubl': (border, border),  # upper baseline left
        'ubr': (border, border + court_width),  # upper baseline right
        'u3l': (border + corner_3pt_from_baseline, border + corner_3pt_from_sideline),  # upper 3pt left
        'u3r': (border + corner_3pt_from_baseline, border + court_width - corner_3pt_from_sideline),  # upper 3pt right
        'u3p': (border + peak_distance, border + court_width//2),  # upper 3pt peak
        'u3bl': (border, border + corner_3pt_from_sideline),  # upper 3pt baseline left
        'u3br': (border, border + court_width - corner_3pt_from_sideline),  # upper 3pt baseline right
        'ukl': (border + key_height, border + (court_width - key_width)//2),  # upper key left
        'ukr': (border + key_height, border + (court_width + key_width)//2),  # upper key right
        'cc': (center_x, center_y),  # center court
        'ml': (center_x, border),  # midline left
        'mr': (center_x, border + court_width),  # midline right
        'lbl': (border + court_length, border),  # lower baseline left
        'lbr': (border + court_length, border + court_width),  # lower baseline right
        'l3l': (border + court_length - corner_3pt_from_baseline, border + corner_3pt_from_sideline),  # lower 3pt left
        'l3r': (border + court_length - corner_3pt_from_baseline, border + court_width - corner_3pt_from_sideline),  # lower 3pt right
        'l3p': (border + court_length - peak_distance, border + court_width//2),  # lower 3pt peak
        'l3bl': (border + court_length, border + corner_3pt_from_sideline),  # lower 3pt baseline left
        'l3br': (border + court_length, border + court_width - corner_3pt_from_sideline),  # lower 3pt baseline right
        'lkl': (border + court_length - key_height, border + (court_width - key_width)//2),  # lower key left
        'lkr': (border + court_length - key_height, border + (court_width + key_width)//2),  # lower key right
        'ub': (border + basket_offset, border + court_width//2),  # upper basket
        'lb': (border + court_length - basket_offset, border + court_width//2),  # lower basket
        'ubml': (border, border + baseline_mark_distance),  # upper baseline mark left
        'ubmr': (border, border + court_width - baseline_mark_distance),  # upper baseline mark right
        'lbml': (border + court_length, border + baseline_mark_distance),  # lower baseline mark left
        'lbmr': (border + court_length, border + court_width - baseline_mark_distance),  # lower baseline mark right
        'cct': (center_x, center_y - circle_radius),  # center circle top
        'ccb': (center_x, center_y + circle_radius),  # center circle bottom
        'ccl': (center_x - circle_radius, center_y),  # center circle left
        'ccr': (center_x + circle_radius, center_y),  # center circle right
        'usml': (border + sideline_mark_distance, border + sideline_mark_inset),  # upper sideline mark left
        'usmr': (border + court_length - sideline_mark_distance, border + sideline_mark_inset),  # upper sideline mark right
        'usml_side': (border + sideline_mark_distance, border),  # upper sideline mark left intersection
        'usmr_side': (border + court_length - sideline_mark_distance, border),  # upper sideline mark right intersection
        'up1': (border + int(48/12 * 10), border + court_width - baseline_mark_distance - int(30/12 * 10)),  # upper point 1
        'up2': (border + int(48/12 * 10), border + baseline_mark_distance + int(30/12 * 10)),  # upper point 2
        'lp1': (border + court_length - int(48/12 * 10), border + court_width - baseline_mark_distance - int(30/12 * 10)),  # lower point 1
        'lp2': (border + court_length - int(48/12 * 10), border + baseline_mark_distance + int(30/12 * 10))  # lower point 2
    }

    # Add corner markers with 3'1" offset and 12" inward distance
    # Upper Right quadrant
    points['ur_outer'] = (points['ubmr'][0] + inward_dist, points['ubmr'][1] + dist_from_baseline)
    points['ur_inner'] = (points['ubmr'][0], points['ubmr'][1] + dist_from_baseline)

    # Upper Left quadrant
    points['ul_outer'] = (points['ubml'][0] + inward_dist, points['ubml'][1] - dist_from_baseline)
    points['ul_inner'] = (points['ubml'][0], points['ubml'][1] - dist_from_baseline)

    # Lower Right quadrant
    points['lr_outer'] = (points['lbmr'][0] - inward_dist, points['lbmr'][1] + dist_from_baseline)
    points['lr_inner'] = (points['lbmr'][0], points['lbmr'][1] + dist_from_baseline)

    # Lower Left quadrant
    points['ll_outer'] = (points['lbml'][0] - inward_dist, points['lbml'][1] - dist_from_baseline)
    points['ll_inner'] = (points['lbml'][0], points['lbml'][1] - dist_from_baseline)

    # Add the new baseline mark points
    for i, distance in enumerate(additional_marks):
        # Upper baseline mark points with line depth
        y_base = border + court_width - baseline_mark_distance
        points[f'ubm_right_{i}'] = (border + distance, y_base + line_depth)
        points[f'ubm_right_base_{i}'] = (border + distance, y_base)
        
        y_base = border + baseline_mark_distance
        points[f'ubm_left_{i}'] = (border + distance, y_base - line_depth)
        points[f'ubm_left_base_{i}'] = (border + distance, y_base)
        
        # Lower baseline mark points with line depth
        y_base = border + court_width - baseline_mark_distance
        points[f'lbm_right_{i}'] = (border + court_length - distance, y_base + line_depth)
        points[f'lbm_right_base_{i}'] = (border + court_length - distance, y_base)
        
        y_base = border + baseline_mark_distance
        points[f'lbm_left_{i}'] = (border + court_length - distance, y_base - line_depth)
        points[f'lbm_left_base_{i}'] = (border + court_length - distance, y_base)
    
    return points, circle_radius, three_point_radius

def transform_point(point, homography):
    """
    Transform a point using homography matrix.
    
    Args:
        point: Tuple of (x, y) coordinates
        homography: 3x3 homography matrix
    
    Returns:
        Tuple of transformed (x, y) coordinates
    """
    # Convert to homogeneous coordinates
    x, y = point
    p = np.array([x, y, 1.0])
    
    # Apply homography
    p_transformed = homography @ p
    
    # Convert back from homogeneous coordinates
    x_transformed = p_transformed[0] / p_transformed[2]
    y_transformed = p_transformed[1] / p_transformed[2]
    
    return (int(round(x_transformed)), int(round(y_transformed)))

def generate_perspective_aware_grid_points(court_width, court_length, border, grid_size, w0_meters=1.75):
    """Generate grid points with perspective-aware sampling."""
    available_height = court_width - 2 * border
    available_width = court_length - 2 * border
    n_rows = grid_size[0]
    n_cols = grid_size[1]
    points = []

    # Generate y-coordinates (rows) with perspective-aware spacing
    if n_rows < 2:
        y_coords = [border + available_height // 2]
    else:
        base = 1.5  # Controls how quickly points spread out
        y_coords = []
        for i in range(n_rows):
            t = i / (n_rows - 1)  # goes from 0 to 1
            y_normalized = (pow(base, t) - 1) / (base - 1)
            y = int(round(border + y_normalized * available_height))
            y_coords.append(y)

    # Generate x-coordinates (columns) with even spacing
    if n_cols < 2:
        x_coords = [border + available_width // 2]
    else:
        x_step = available_width / (n_cols - 1)
        x_coords = [int(round(border + col_idx * x_step)) for col_idx in range(n_cols)]

    # Build the full grid
    for y in y_coords:
        for x in x_coords:
            points.append((x, y))

    return points
