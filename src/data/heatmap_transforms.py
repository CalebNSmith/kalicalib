# In src/data/heatmap_transforms.py

import torchvision.transforms as transforms
import torch
import random
import cv2
import numpy as np

class ColorJitter:
    def __init__(self, brightness=0.7, contrast=0.5, saturation=0.5, hue=0.5):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, image):
        # Ensure image is in correct format for torchvision transforms
        if isinstance(image, np.ndarray):
            # Convert from HWC to CHW format
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float() / 255.0
        
        # Apply color jitter
        image = self.color_jitter(image)
        
        # Convert back to numpy array in HWC format
        if isinstance(image, torch.Tensor):
            image = (image.numpy() * 255).astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))
        
        return image

class RandomHorizontalFlip:
    """Apply random horizontal flipping to image and heatmaps."""
    def __init__(self, p=0.5):
        self.p = p
        self.grid_width = 13
        self.grid_height = 7
        
        # Pre-compute keypoint remapping indices for efficiency
        self.keypoint_mapping = {}
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                old_idx = y * self.grid_width + x
                new_x = self.grid_width - 1 - x  # Flip x coordinate
                new_idx = y * self.grid_width + new_x
                self.keypoint_mapping[old_idx] = new_idx

    def __call__(self, image, target=None):
        if random.random() >= self.p:
            return image, target

        # Ensure image is numpy array in HWC format
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:  # CHW format
                image = image.permute(1, 2, 0)
            image = image.numpy()

        # Flip image
        image = cv2.flip(image, 1)  # Horizontal flip

        if target is None:
            return image, None

        # Handle homography matrix
        if isinstance(target, np.ndarray) and target.shape == (3, 3):
            flip_matrix = np.array([[-1, 0, image.shape[1]], 
                                  [0, 1, 0], 
                                  [0, 0, 1]])
            target = flip_matrix @ target @ np.linalg.inv(flip_matrix)
            return image, target

        # Handle heatmap tensor
        if isinstance(target, (torch.Tensor, np.ndarray)):
            is_torch = isinstance(target, torch.Tensor)
            if is_torch:
                target = target.numpy()
            
            target = np.flip(target, axis=2)  # Flip spatially
            remapped = np.zeros_like(target)

            # Remap grid points (0-90)
            for old_idx, new_idx in self.keypoint_mapping.items():
                remapped[new_idx] = target[old_idx]

            # Copy over basket points (no index remapping needed)
            remapped[91] = target[91]  # ub
            remapped[92] = target[92]  # lb
            
            # Copy background channel
            remapped[-1] = target[-1]  # Use -1 to handle both 93 and 94 channel cases

            if is_torch:
                remapped = torch.from_numpy(remapped)

            return image, remapped

        raise ValueError(f"Unsupported target type: {type(target)}")