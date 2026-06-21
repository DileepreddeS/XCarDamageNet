"""Image preprocessing utilities for XCarDamageNet.

Handles resize, normalisation, and format conversion for inference.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from typing import Tuple, Union


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(
    image: Union[str, np.ndarray],
    img_size: int = 518,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load, resize, and normalise an image for XCarDamageNet inference.

    Args:
        image: File path string or BGR numpy array from cv2.imread.
        img_size: Target size (square).

    Returns:
        tensor:        (1, 3, img_size, img_size) — batch-ready tensor
        original_size: (H, W) of original image for coordinate denormalisation
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise IOError(f"Cannot load image: {image}")
    else:
        img = image.copy()

    original_size = (img.shape[0], img.shape[1])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD

    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)
    return tensor, original_size


def denormalise_boxes(
    boxes: torch.Tensor,
    original_size: Tuple[int, int],
) -> torch.Tensor:
    """Convert normalised [0,1] boxes to pixel coordinates.

    Args:
        boxes:         (N, 4) in [x1, y1, x2, y2] normalised
        original_size: (H, W) of original image

    Returns:
        (N, 4) pixel coordinates
    """
    H, W = original_size
    scale = torch.tensor([W, H, W, H], dtype=boxes.dtype, device=boxes.device)
    return boxes * scale
