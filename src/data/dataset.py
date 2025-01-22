# File: src/data/dataset.py
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

class KaliCalibDataset(Dataset):
    """Dataset class for KaliCalib training data"""

    def __init__(self,
                data_dir: str,
                data_prep,
                transform=None,
                split: str = 'train'):
        """
        Args:
            data_dir: Root directory of dataset
            data_prep: Instance of KaliCalibDataPrep
            transform: Optional transform to be applied
            split: One of ['train', 'val', 'test']
        """
        self.data_dir = Path(data_dir)
        self.data_prep = data_prep
        self.transform = transform
        self.split = split

        # Get all image paths
        self.image_dir = self.data_dir / 'images'
        self.label_dir = self.data_dir / 'labels'

        self.samples = []
        for label_path in self.label_dir.glob('*.npz'):
            image_path = self.image_dir / f"{label_path.stem}.jpg"
            if image_path.exists():
                self.samples.append((str(image_path), str(label_path)))

        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 samples in {data_dir}")

        # Print dataset info
        print(f"Found {len(self.samples)} samples in {data_dir}")
        if len(self.samples) > 0:
            print(f"First sample - Image: {self.samples[0][0]}, Label: {self.samples[0][1]}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[idx]

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Load label data from NPZ file
        try:
            with np.load(label_path, allow_pickle=True) as npz_data:
                # -- Now we have:
                #    - 91 grid points: grid_0 ... grid_90
                #    - ub (upper basket)
                #    - lb (lower basket)
                #    - background
                #
                # => total "foreground" channels = 91 + 2 = 93
                # => plus 1 background = 94 total channels

                n_keypoints = 93  # 91 grids + ub + lb
                heatmap_h = image.shape[0] // self.data_prep.output_stride
                heatmap_w = image.shape[1] // self.data_prep.output_stride

                # Initialize at 1/4 resolution
                # Channels layout:
                #   indices [0..90]   -> grid_0..grid_90  (91 channels)
                #   index 91         -> ub
                #   index 92         -> lb
                #   index 93         -> background
                heatmap = np.zeros((n_keypoints + 1, heatmap_h, heatmap_w), dtype=np.float32)

                # 1. Load grid channels
                for i in range(91):
                    key = f'grid_{i}'
                    if key not in npz_data:
                        raise KeyError(f"Missing grid point {i} in label file")
                    grid = npz_data[key]
                    grid_resized = cv2.resize(grid, (heatmap_w, heatmap_h))
                    heatmap[i] = grid_resized  # i in [0..90]

                # 2. Load ub (upper basket) into index 91
                if 'ub' in npz_data:
                    ub = npz_data['ub']
                    ub_resized = cv2.resize(ub, (heatmap_w, heatmap_h))
                    heatmap[91] = ub_resized

                # 3. Load lb (lower basket) into index 92
                if 'lb' in npz_data:
                    lb = npz_data['lb']
                    lb_resized = cv2.resize(lb, (heatmap_w, heatmap_h))
                    heatmap[92] = lb_resized

                # 4. Load background into index 93 (the last channel)
                if 'background' in npz_data:
                    bg = npz_data['background']
                    bg_resized = cv2.resize(bg, (heatmap_w, heatmap_h))
                    heatmap[93] = bg_resized

        except Exception as e:
            raise ValueError(f"Failed to load label data from {label_path}: {str(e)}")

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)
        else:
            # If no transform is provided, convert to tensor here
            # shape => (C, H, W)
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

        # Convert heatmap to a PyTorch tensor
        heatmap = torch.from_numpy(heatmap).float()

        return image, heatmap