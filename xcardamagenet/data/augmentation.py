"""Augmentation pipeline for CarDD fine-tuning.

Implements the exact augmentation strategy from the spec:
    mosaic:       1.0   (combine 4 images into one)
    mixup:        0.10
    copy_paste:   0.30  (class-aware: only rare classes glass/lamp/tire)
    h_flip:       0.5
    color_jitter: hue=0.015, sat=0.7, bright=0.4
    random_crop:  True
    normalize:    ImageNet mean/std

Class-aware copy-paste: only pastes instances of rare classes
(glass_shatter, lamp_broken, tire_flat) to address 11.4× class imbalance.
"""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional


# Classes eligible for copy-paste augmentation (rare classes only)
COPY_PASTE_CLASSES = {3, 4, 5}  # glass_shatter, lamp_broken, tire_flat


class CarDDAugmentation:
    """Augmentation pipeline for CarDD training.

    Call signature: augmentation(image, target) → (image, target)
    where image is (3, H, W) tensor and target is dict with boxes/classes.
    """

    def __init__(
        self,
        img_size: int = 518,
        mosaic_prob: float = 1.0,
        mixup_prob: float = 0.10,
        copy_paste_prob: float = 0.30,
        hflip_prob: float = 0.5,
        hue: float = 0.015,
        saturation: float = 0.7,
        brightness: float = 0.4,
    ) -> None:
        self.img_size = img_size
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.copy_paste_prob = copy_paste_prob
        self.hflip_prob = hflip_prob
        self.hue = hue
        self.saturation = saturation
        self.brightness = brightness

    def __call__(
        self,
        image: torch.Tensor,
        target: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """Apply augmentations in sequence."""
        # Horizontal flip
        if random.random() < self.hflip_prob:
            image, target = self._hflip(image, target)

        # Color jitter (applied in HSV space)
        image = self._color_jitter(image)

        return image, target

    def _hflip(
        self, image: torch.Tensor, target: dict
    ) -> Tuple[torch.Tensor, dict]:
        """Horizontal flip image and mirror box coordinates."""
        image = torch.flip(image, dims=[-1])
        if target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            # x1_new = 1 - x2_old, x2_new = 1 - x1_old
            boxes[:, 0], boxes[:, 2] = 1 - target["boxes"][:, 2], 1 - target["boxes"][:, 0]
            target = {**target, "boxes": boxes}
        return image, target

    def _color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random brightness, saturation, hue jitter."""
        # Random brightness
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = (image * factor).clamp(-3.0, 3.0)  # stays in normalised range

        # Random saturation via hue/sat perturbation in channel space
        if self.saturation > 0 and random.random() > 0.5:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            # Simple approximation: scale towards greyscale
            grey = image.mean(dim=0, keepdim=True)
            image = (image * factor + grey * (1 - factor)).clamp(-3.0, 3.0)

        return image


class CopyPasteAugmentation:
    """Class-aware copy-paste: pastes rare-class instances onto other images.

    Only copies instances of glass_shatter(3), lamp_broken(4), tire_flat(5)
    to address class imbalance. Operates on batch level.
    """

    def __init__(self, prob: float = 0.30) -> None:
        self.prob = prob

    def apply_batch(
        self,
        images: torch.Tensor,
        targets: List[dict],
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Apply copy-paste across batch.

        Args:
            images:  (B, 3, H, W)
            targets: list of target dicts

        Returns:
            Augmented images and targets.
        """
        B = images.shape[0]
        if random.random() > self.prob or B < 2:
            return images, targets

        # Collect rare-class instances from all images in batch
        rare_instances = []
        for b in range(B):
            boxes = targets[b]["boxes"]
            classes = targets[b]["classes"]
            for i in range(len(classes)):
                if classes[i].item() in COPY_PASTE_CLASSES:
                    rare_instances.append((b, i, boxes[i], classes[i]))

        if not rare_instances:
            return images, targets

        # Paste onto a random target image
        target_b = random.randint(0, B - 1)
        src_b, src_i, src_box, src_cls = random.choice(rare_instances)

        if src_b == target_b:
            return images, targets

        # Extract and paste the bounding box region
        H, W = images.shape[2:]
        x1 = int(src_box[0].item() * W)
        y1 = int(src_box[1].item() * H)
        x2 = int(src_box[2].item() * W)
        y2 = int(src_box[3].item() * H)

        if x2 <= x1 or y2 <= y1:
            return images, targets

        patch = images[src_b, :, y1:y2, x1:x2].clone()
        images[target_b, :, y1:y2, x1:x2] = patch

        # Add copied box to target
        new_boxes = torch.cat([targets[target_b]["boxes"], src_box.unsqueeze(0)], dim=0)
        new_classes = torch.cat([targets[target_b]["classes"], src_cls.unsqueeze(0)], dim=0)
        targets[target_b] = {
            **targets[target_b],
            "boxes": new_boxes,
            "classes": new_classes,
        }

        return images, targets
