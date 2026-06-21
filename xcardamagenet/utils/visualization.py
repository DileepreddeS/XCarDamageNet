"""Visualization utilities for XCarDamageNet predictions.

Draws bounding boxes, class labels, severity scores, and attention heatmaps.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from typing import List, Optional


CLASS_NAMES = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]
COLORS = [
    (255, 100, 100),  # dent — red
    (100, 255, 100),  # scratch — green
    (100, 100, 255),  # crack — blue
    (255, 255, 100),  # glass_shatter — yellow
    (255, 100, 255),  # lamp_broken — magenta
    (100, 255, 255),  # tire_flat — cyan
]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    severity: Optional[float] = None,
    cause: Optional[str] = None,
    repair: Optional[str] = None,
    fraud_risk: Optional[float] = None,
) -> np.ndarray:
    """Draw detection results on image.

    Args:
        image: (H, W, 3) BGR image
        boxes: (N, 4) pixel coordinates [x1,y1,x2,y2]
        classes: (N,) class ids
        scores: (N,) confidence scores
        severity: optional severity score
        cause: optional cause string
        repair: optional repair action string
        fraud_risk: optional fraud probability

    Returns:
        Annotated image (H, W, 3) BGR
    """
    img = image.copy()

    for i, (box, cls_id, score) in enumerate(zip(boxes, classes, scores)):
        x1, y1, x2, y2 = box.astype(int)
        color = COLORS[cls_id % len(COLORS)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Summary info
    y_offset = 20
    if severity is not None:
        cv2.putText(img, f"Severity: {severity:.2f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    if cause:
        cv2.putText(img, f"Cause: {cause}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    if repair:
        cv2.putText(img, f"Repair: {repair}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    if fraud_risk is not None and fraud_risk > 0.5:
        cv2.putText(img, f"FRAUD RISK: {fraud_risk:.2f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return img


def draw_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay attention heatmap on image.

    Args:
        image:   (H, W, 3) BGR image
        heatmap: (H', W') float array in [0, 1]
        alpha:   blend factor for overlay
        colormap: cv2 colormap

    Returns:
        Overlay image (H, W, 3) BGR
    """
    H, W = image.shape[:2]
    hmap = cv2.resize(heatmap, (W, H))
    hmap = (hmap * 255).astype(np.uint8)
    hmap_color = cv2.applyColorMap(hmap, colormap)
    return cv2.addWeighted(image, 1 - alpha, hmap_color, alpha, 0)
