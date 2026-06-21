"""CarDD Dataset loader for XCarDamageNet.

CarDD is in YOLO format:
    images/  — JPEG/PNG images
    labels/  — .txt files, one per image, each line: class x_c y_c w h (normalised)

Splits: train=2816 images, val=810, test=374
Classes: 0=dent, 1=scratch, 2=crack, 3=glass_shatter, 4=lamp_broken, 5=tire_flat
Average resolution: 979×705
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable, Tuple


CLASS_NAMES = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]
NUM_CLASSES = len(CLASS_NAMES)


class CarDDDataset(Dataset):
    """PyTorch Dataset for CarDD in YOLO format.

    Returns:
        image:  (3, H, W) normalised tensor
        target: dict with keys:
            "boxes":   (N, 4) float32 in [x1, y1, x2, y2] normalised [0,1]
            "classes": (N,) int64 class ids
            "image_id": str — filename stem
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 518,
        transforms: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            root: Path to CarDD dataset root. Expected structure:
                root/images/{split}/  and  root/labels/{split}/
            split: 'train', 'val', or 'test'
            img_size: Resize images to this size (square).
            transforms: Optional augmentation callable(image, target) → (image, target).
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.transforms = transforms

        img_dir = self.root / "images" / split
        lbl_dir = self.root / "labels" / split

        if not img_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {img_dir}")

        # Find all image files
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_paths = sorted([
            p for p in img_dir.iterdir()
            if p.suffix.lower() in extensions
        ])

        # Map each image to its label file (may not exist = no annotations)
        self.label_paths = []
        for img_path in self.image_paths:
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            self.label_paths.append(lbl_path if lbl_path.exists() else None)

        # ImageNet normalisation parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]

        # Load and resize image (BGR → RGB)
        img = cv2.imread(str(img_path))
        if img is None:
            raise IOError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Normalise to [0, 1] then apply ImageNet mean/std
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))  # (3, H, W)

        # Parse YOLO-format labels
        boxes, classes = self._load_labels(lbl_path)

        target = {
            "boxes": boxes,
            "classes": classes,
            "image_id": img_path.stem,
        }

        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target

    def _load_labels(
        self, lbl_path: Optional[Path]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse YOLO-format label file.

        YOLO format per line: class x_center y_center width height (normalised)
        Returns boxes in [x1, y1, x2, y2] normalised format.
        """
        if lbl_path is None or not lbl_path.exists():
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.int64)

        boxes, classes = [], []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # Convert centre format to corner format
                x1 = x_c - w / 2
                y1 = y_c - h / 2
                x2 = x_c + w / 2
                y2 = y_c + h / 2
                boxes.append([x1, y1, x2, y2])
                classes.append(cls_id)

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.int64)

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(classes, dtype=torch.int64),
        )

    @staticmethod
    def collate_fn(batch: list) -> Tuple[torch.Tensor, list]:
        """Collate function for DataLoader — handles variable-length label lists."""
        images = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]
        return images, targets
