"""Unlabeled car image dataset for MAE physics pre-training.

Used to pre-train the PhysicsTokenEncoder via Masked Autoencoder (MAE).
No annotations required — self-supervised via reconstruction.

Data sources (free, no license restrictions):
    - Stanford Cars (~16K images)
    - CompCars (~30K images)
    - COCO car crops (~12K extracted images)
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable


class UnlabeledCarDataset(Dataset):
    """Dataset for self-supervised pre-training on unlabeled car images.

    Loads images from a flat directory, applies basic augmentation,
    and returns (original_image, masked_image, mask) for MAE training.
    """

    def __init__(
        self,
        image_dir: str,
        img_size: int = 518,
        mask_ratio: float = 0.75,
        transforms: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            image_dir: Directory containing unlabeled car images.
            img_size: Resize to this size.
            mask_ratio: Fraction of patches to mask for MAE. Default 0.75.
            transforms: Optional augmentation callable.
        """
        super().__init__()
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        self.transforms = transforms

        img_dir = Path(image_dir)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_paths = sorted([
            p for p in img_dir.rglob("*")
            if p.suffix.lower() in extensions
        ])

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Patch grid dimensions for masking
        self.patch_size = 14  # DINOv2 patch size
        self.grid_size = img_size // self.patch_size  # e.g. 37 for 518

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            # Return zeros for corrupted images (log is captured during training)
            return (
                torch.zeros(3, self.img_size, self.img_size),
                torch.zeros(3, self.img_size, self.img_size),
                torch.zeros(self.grid_size * self.grid_size, dtype=torch.bool),
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))  # (3, H, W)

        if self.transforms is not None:
            img_tensor = self.transforms(img_tensor)

        # Generate random patch mask for MAE
        n_patches = self.grid_size * self.grid_size
        n_masked = int(n_patches * self.mask_ratio)
        mask = torch.zeros(n_patches, dtype=torch.bool)
        mask[torch.randperm(n_patches)[:n_masked]] = True

        # Create masked image (set masked patches to 0)
        masked_img = img_tensor.clone()
        mask_2d = mask.view(self.grid_size, self.grid_size)
        mask_resized = mask_2d.float().unsqueeze(0).unsqueeze(0)
        mask_resized = torch.nn.functional.interpolate(
            mask_resized, size=(self.img_size, self.img_size), mode="nearest"
        ).squeeze()
        masked_img = img_tensor * (1 - mask_resized)

        return img_tensor, masked_img, mask  # (original, masked, patch_mask)
