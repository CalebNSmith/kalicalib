# src/data/data_prep.py
import cv2
import numpy as np
import torch
from torchvision import transforms
import math
from ..utils.court import calculate_court_points

class KaliCalibDataPrep:
    def __init__(self, config):
        self.input_size = tuple(config['model']['input_size'])  # width, height
        self.output_stride = config['model'].get('output_stride', 4)
        self.output_size = (self.input_size[0] // self.output_stride,
                           self.input_size[1] // self.output_stride)
        self.disk_radius = config['data']['disk_radius']
        
        # Court dimensions
        self.court_width = config['data']['court_width']
        self.court_length = config['data']['court_length']
        self.border = config['data']['border']
        self.grid_size = tuple(config['data']['grid_size'])
        self.n_points = self.grid_size[0] * self.grid_size[1]

        # Calculate court points once during initialization
        self._court_points, self._circle_radius, self._three_point_radius = calculate_court_points(
            self.court_width,
            self.court_length,
            self.border
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def get_court_points(self):
        """Return the cached court points and related dimensions."""
        return self._court_points, self._circle_radius, self._three_point_radius

    def create_disk_mask(self, center, radius):
        """Create a disk-shaped mask centered at the given point."""
        y, x = np.ogrid[:self.output_size[1], :self.output_size[0]]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return (dist <= radius).astype(np.float32)

    def generate_heatmaps(self, image, npz_data):
        """Generate heatmaps for keypoints, bounds, and background."""
        if image.shape[:2] != (self.input_size[1], self.input_size[0]):
            image = cv2.resize(image, self.input_size)

        normalized_image = self.transform(image)

        # Initialize heatmaps array: 91 grid points + ub + lb + background = 94 channels
        heatmaps = np.zeros((self.n_points + 3, self.output_size[1], self.output_size[0]),
                        dtype=np.float32)

        # Load all grid points (0-90)
        for i in range(91):
            grid_key = f'grid_{i}'
            if grid_key in npz_data:
                heatmaps[i] = cv2.resize(npz_data[grid_key], 
                                    (self.output_size[0], self.output_size[1]),
                                    interpolation=cv2.INTER_AREA)

        # Load upper and lower baskets at indices 91 and 92
        if 'ub' in npz_data:
            heatmaps[91] = cv2.resize(npz_data['ub'],
                                    (self.output_size[0], self.output_size[1]),
                                    interpolation=cv2.INTER_AREA)
        if 'lb' in npz_data:
            heatmaps[92] = cv2.resize(npz_data['lb'],
                                    (self.output_size[0], self.output_size[1]),
                                    interpolation=cv2.INTER_AREA)

        # Load background at index 93
        if 'background' in npz_data:
            heatmaps[93] = cv2.resize(npz_data['background'],
                                    (self.output_size[0], self.output_size[1]),
                                    interpolation=cv2.INTER_AREA)

        return normalized_image, torch.from_numpy(heatmaps)

    def transform_point(self, point, homography):
        """Transform a point using homography matrix."""
        x, y = point
        p = np.array([x, y, 1.0])
        p_transformed = homography @ p
        x_transformed = p_transformed[0] / p_transformed[2]
        y_transformed = p_transformed[1] / p_transformed[2]
        return (int(round(x_transformed)), int(round(y_transformed)))

    def generate_grid_points(self, w0=32):
        """Generate grid points for keypoint detection."""
        # Available space for sampling
        available_height = self.court_width - 2 * self.border
        available_width = self.court_length - 2 * self.border

        # Number of rows and columns
        N_rows = self.grid_size[0]
        N_cols = self.grid_size[1]

        points = []

        # Generate y-coordinates (rows)
        if N_rows < 2:
            y_coords = [int(round(self.court_width - self.border))]
        else:
            y_step = available_height / (N_rows - 1)
            y_coords = [
                int(round(self.border + row_idx * y_step))
                for row_idx in range(N_rows)
            ]

        # Generate x-coordinates (columns)
        if N_cols < 2:
            x_coords = [int(round(self.border + available_width / 2))]
        else:
            x_step = available_width / (N_cols - 1)
            x_coords = [
                int(round(self.border + col_idx * x_step))
                for col_idx in range(N_cols)
            ]

        # Build the full grid
        for y in y_coords:
            for x in x_coords:
                points.append((x, y))

        return points