"""Shape-Aware CIoU Loss — primary box regression loss for XCarDamageNet.

Standard CIoU + per-class shape penalty using WIDTH/HEIGHT RATIOS (always [0,1]).

CRITICAL: Shape penalty MUST use ratios, not absolute coordinates. We attempted
to use absolute values in XCarDamage v1 FIVE times — box_loss exploded to 2000
every time because YOLO-style coordinates are stride-normalised, not in [0,1].
Ratios are scale-invariant and always produce stable training.

Formula:
    omega_w = |pred_w - gt_w| / max(pred_w, gt_w)      # [0, 1]
    omega_h = |pred_h - gt_h| / max(pred_h, gt_h)      # [0, 1]
    shape_penalty = (1 - exp(-omega_w))^4 + (1 - exp(-omega_h))^4
    L_box = (1 - CIoU + class_weight * shape_penalty) * cb_weight

Per-class penalty weights (thin/elongated shapes penalised more):
    dent=0.03, scratch=0.08, crack=0.10, glass=0.02, lamp=0.04, tire=0.03
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


# Per-class shape penalty weights: [dent, scratch, crack, glass, lamp, tire]
_SHAPE_WEIGHTS = torch.tensor([0.03, 0.08, 0.10, 0.02, 0.04, 0.03], dtype=torch.float32)


def _ciou(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute standard CIoU between predicted and ground-truth boxes.

    Args:
        pred: (N, 4) boxes in [x1, y1, x2, y2] format
        gt:   (N, 4) boxes in [x1, y1, x2, y2] format

    Returns:
        ciou: (N,) per-box CIoU in [-1, 1]
    """
    # Intersection
    inter_x1 = torch.max(pred[:, 0], gt[:, 0])
    inter_y1 = torch.max(pred[:, 1], gt[:, 1])
    inter_x2 = torch.min(pred[:, 2], gt[:, 2])
    inter_y2 = torch.min(pred[:, 3], gt[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    pred_w = (pred[:, 2] - pred[:, 0]).clamp(min=eps)
    pred_h = (pred[:, 3] - pred[:, 1]).clamp(min=eps)
    gt_w = (gt[:, 2] - gt[:, 0]).clamp(min=eps)
    gt_h = (gt[:, 3] - gt[:, 1]).clamp(min=eps)

    pred_area = pred_w * pred_h
    gt_area = gt_w * gt_h
    union_area = pred_area + gt_area - inter_area
    iou = inter_area / (union_area + eps)

    # Enclosing box diagonal squared
    enc_x1 = torch.min(pred[:, 0], gt[:, 0])
    enc_y1 = torch.min(pred[:, 1], gt[:, 1])
    enc_x2 = torch.max(pred[:, 2], gt[:, 2])
    enc_y2 = torch.max(pred[:, 3], gt[:, 3])
    c_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # Centre distance squared
    pred_cx = (pred[:, 0] + pred[:, 2]) / 2
    pred_cy = (pred[:, 1] + pred[:, 3]) / 2
    gt_cx = (gt[:, 0] + gt[:, 2]) / 2
    gt_cy = (gt[:, 1] + gt[:, 3]) / 2
    centre_dist_sq = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2

    # Aspect ratio consistency term (v)
    v = (4 / (torch.pi ** 2)) * (
        torch.atan(gt_w / gt_h) - torch.atan(pred_w / pred_h)
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - centre_dist_sq / c_diag_sq - alpha * v
    return ciou


def _shape_penalty(
    pred_w: torch.Tensor,
    pred_h: torch.Tensor,
    gt_w: torch.Tensor,
    gt_h: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute scale-invariant shape penalty using width/height RATIOS.

    omega_w and omega_h are always in [0, 1], guaranteeing stable training
    regardless of coordinate normalisation scheme.
    """
    omega_w = (pred_w - gt_w).abs() / (torch.max(pred_w, gt_w) + eps)  # [0, 1]
    omega_h = (pred_h - gt_h).abs() / (torch.max(pred_h, gt_h) + eps)  # [0, 1]
    penalty = (1 - torch.exp(-omega_w)) ** 4 + (1 - torch.exp(-omega_h)) ** 4
    return penalty


class ShapeAwareCIoULoss(nn.Module):
    """CIoU loss augmented with per-class shape penalty using w/h ratios.

    Handles class-balanced weighting (cb_weight) externally passed per sample
    to integrate with ClassBalancedBCELoss's effective number of samples logic.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        self.register_buffer("shape_weights", _SHAPE_WEIGHTS)

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        class_ids: torch.Tensor,
        cb_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred:       (N, 4) predicted boxes [x1, y1, x2, y2]
            gt:         (N, 4) ground-truth boxes [x1, y1, x2, y2]
            class_ids:  (N,) integer class indices [0-5]
            cb_weights: (N,) class-balanced sample weights. If None, all=1.

        Returns:
            loss: scalar (mean/sum) or (N,) per-sample (none)
        """
        if pred.numel() == 0:
            return pred.sum() * 0  # empty tensor, 0 loss with gradient

        eps = 1e-7
        ciou = _ciou(pred, gt, eps)  # (N,)

        # Width/height in absolute coords for ratio computation
        pred_w = (pred[:, 2] - pred[:, 0]).clamp(min=eps)
        pred_h = (pred[:, 3] - pred[:, 1]).clamp(min=eps)
        gt_w = (gt[:, 2] - gt[:, 0]).clamp(min=eps)
        gt_h = (gt[:, 3] - gt[:, 1]).clamp(min=eps)

        penalty = _shape_penalty(pred_w, pred_h, gt_w, gt_h, eps)  # (N,)

        # Per-class shape penalty weight
        cls_w = self.shape_weights[class_ids.clamp(0, 5)]  # (N,)

        # Class-balanced sample weights
        if cb_weights is None:
            cb_weights = torch.ones_like(ciou)

        loss = (1 - ciou + cls_w * penalty) * cb_weights  # (N,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
