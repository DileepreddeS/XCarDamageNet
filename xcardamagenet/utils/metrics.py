"""mAP computation for XCarDamageNet evaluation.

Computes mAP@0.5 and per-class AP using the standard PASCAL VOC 11-point
interpolation method.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import List, Dict


CLASS_NAMES = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) in [x1, y1, x2, y2]
        boxes2: (M, 4) in [x1, y1, x2, y2]

    Returns:
        iou: (N, M) pairwise IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union_area = area1[:, None] + area2[None, :] - inter_area
    return inter_area / (union_area + 1e-7)


def compute_ap(
    recall: np.ndarray, precision: np.ndarray
) -> float:
    """Compute AP using 11-point interpolation (PASCAL VOC)."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.any(recall >= t):
            ap += np.max(precision[recall >= t])
    return ap / 11.0


def compute_ap_per_class(
    predictions: List[dict],
    targets: List[dict],
    num_classes: int = 6,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute AP for each class.

    Args:
        predictions: list of {"boxes": (N,4), "scores": (N,), "classes": (N,)}
        targets:     list of {"boxes": (M,4), "classes": (M,)}
        num_classes: number of classes
        iou_threshold: IoU threshold for TP/FP assignment

    Returns:
        dict with AP per class
    """
    ap_per_class = {}

    for cls_id in range(num_classes):
        all_scores, all_tp, all_fp = [], [], []
        n_gt = 0

        for pred, tgt in zip(predictions, targets):
            # GT boxes for this class
            gt_mask = (tgt["classes"] == cls_id)
            gt_boxes = tgt["boxes"][gt_mask]
            n_gt += gt_mask.sum().item()

            # Predicted boxes for this class
            pred_mask = (pred["classes"] == cls_id)
            if pred_mask.sum() == 0:
                continue

            p_boxes = pred["boxes"][pred_mask]
            p_scores = pred["scores"][pred_mask]

            # Sort by score descending
            order = p_scores.argsort(descending=True)
            p_boxes = p_boxes[order]
            p_scores = p_scores[order]

            matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

            for pb in p_boxes:
                if len(gt_boxes) == 0:
                    all_fp.append(1)
                    all_tp.append(0)
                    continue

                ious = box_iou(pb.unsqueeze(0), gt_boxes)[0]
                best_iou, best_idx = ious.max(0)

                if best_iou >= iou_threshold and not matched[best_idx]:
                    matched[best_idx] = True
                    all_tp.append(1)
                    all_fp.append(0)
                else:
                    all_tp.append(0)
                    all_fp.append(1)

            all_scores.extend(p_scores.tolist())

        if n_gt == 0:
            ap_per_class[CLASS_NAMES[cls_id]] = 0.0
            continue

        # Cumulative TP/FP → precision/recall curve
        tp_arr = np.array(all_tp, dtype=np.float32)
        fp_arr = np.array(all_fp, dtype=np.float32)
        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(fp_arr)
        recall = cum_tp / max(1, n_gt)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-7)

        ap_per_class[CLASS_NAMES[cls_id]] = compute_ap(recall, precision)

    return ap_per_class


def compute_map(
    predictions: List[dict],
    targets: List[dict],
    num_classes: int = 6,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute mAP@iou_threshold and per-class APs.

    Returns:
        dict with "mAP" and per-class APs.
    """
    per_class = compute_ap_per_class(predictions, targets, num_classes, iou_threshold)
    map_score = np.mean(list(per_class.values()))
    return {"mAP": float(map_score), **per_class}
