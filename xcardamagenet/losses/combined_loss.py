"""Combined Loss — weighted sum of all XCarDamageNet loss components.

Total loss formula from spec:
    L = 7.5*L_box + 0.5*L_cls + 1.5*L_dfl + 0.10*L_attn + 0.05*L_contrast + 0.02*L_physics

Where:
    L_box:      Shape-Aware CIoU (box regression)
    L_cls:      Class-Balanced BCE (classification)
    L_dfl:      Distribution Focal Loss (sub-pixel coordinate precision, standard)
    L_attn:     Attention Supervision BCE (explainability heatmaps)
    L_contrast: Contrastive Triplet (damage vs normal discrimination)
    L_physics:  Physics Consistency CE (fraud/inconsistency detection)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .shape_aware_ciou import ShapeAwareCIoULoss
from .class_balanced import ClassBalancedBCELoss
from .attention_loss import AttentionSupervisionLoss
from .contrastive_loss import ContrastiveTripletLoss
from .physics_loss import PhysicsConsistencyLoss


class DFLoss(nn.Module):
    """Distribution Focal Loss for sub-pixel box coordinate regression.

    Standard DFL from YOLO — proven effective, kept from v1.
    Treats box coordinate as a discrete distribution over [0, reg_max] bins.
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_dist: (N, 4*reg_max) distribution logits per side
            target:    (N, 4) target in fractional bins [0, reg_max-1]

        Returns:
            loss: scalar DFL
        """
        N = pred_dist.shape[0]
        if N == 0:
            return pred_dist.sum() * 0

        # Reshape to (N, 4, reg_max)
        pred = pred_dist.view(N, 4, self.reg_max)
        tgt_left = target.long().clamp(0, self.reg_max - 1)
        tgt_right = (tgt_left + 1).clamp(0, self.reg_max - 1)
        weight_right = target - tgt_left.float()
        weight_left = 1.0 - weight_right

        log_prob = F.log_softmax(pred, dim=-1)

        loss = -(
            log_prob.gather(-1, tgt_left.unsqueeze(-1)).squeeze(-1) * weight_left
            + log_prob.gather(-1, tgt_right.unsqueeze(-1)).squeeze(-1) * weight_right
        )  # (N, 4)
        return loss.mean()


class CombinedLoss(nn.Module):
    """Weighted combination of all XCarDamageNet loss terms.

    Loss weights (from spec):
        L_box:     7.5 — highest weight, box quality is primary objective
        L_cls:     0.5
        L_dfl:     1.5 — sub-pixel box precision
        L_attn:    0.10 — explainability supervision
        L_contrast:0.05 — representation learning
        L_physics: 0.02 — physics consistency (smallest, auxiliary)
    """

    # Loss weights matching spec exactly
    W_BOX = 7.5
    W_CLS = 0.5
    W_DFL = 1.5
    W_ATTN = 0.10
    W_CONTRAST = 0.05
    W_PHYSICS = 0.02

    def __init__(
        self,
        class_counts: Optional[list[int]] = None,
        reg_max: int = 16,
    ) -> None:
        super().__init__()
        self.box_loss = ShapeAwareCIoULoss()
        self.cls_loss = ClassBalancedBCELoss(class_counts)
        self.dfl_loss = DFLoss(reg_max)
        self.attn_loss = AttentionSupervisionLoss()
        self.contrast_loss = ContrastiveTripletLoss()
        self.physics_loss = PhysicsConsistencyLoss()

    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        class_ids: torch.Tensor,
        pred_cls_logits: torch.Tensor,
        gt_cls_targets: torch.Tensor,
        pred_dist: Optional[torch.Tensor] = None,
        dfl_targets: Optional[torch.Tensor] = None,
        attn_maps: Optional[torch.Tensor] = None,
        gt_boxes_list: Optional[list] = None,
        gt_classes_list: Optional[list] = None,
        anchor: Optional[torch.Tensor] = None,
        positive: Optional[torch.Tensor] = None,
        negative: Optional[torch.Tensor] = None,
        physics_implied: Optional[torch.Tensor] = None,
        predicted_class_logits: Optional[torch.Tensor] = None,
        cb_weights: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses and return weighted sum.

        Required:
            pred_boxes:      (N, 4) [x1,y1,x2,y2]
            gt_boxes:        (N, 4) [x1,y1,x2,y2]
            class_ids:       (N,) integer class ids
            pred_cls_logits: (N, num_classes) classification logits
            gt_cls_targets:  (N, num_classes) binary classification targets

        Optional (skip corresponding loss term if None):
            pred_dist/dfl_targets: for DFL
            attn_maps + gt_boxes_list + gt_classes_list: for attention loss
            anchor/positive/negative: for contrastive loss
            physics_implied/predicted_class_logits: for physics loss
            cb_weights: per-sample class-balanced weights for box loss

        Returns:
            dict with keys: total, box, cls, dfl, attn, contrast, physics
        """
        losses = {}

        # L_box
        losses["box"] = self.box_loss(pred_boxes, gt_boxes, class_ids, cb_weights)

        # L_cls
        losses["cls"] = self.cls_loss(pred_cls_logits, gt_cls_targets)

        # L_dfl
        if pred_dist is not None and dfl_targets is not None:
            losses["dfl"] = self.dfl_loss(pred_dist, dfl_targets)
        else:
            losses["dfl"] = pred_boxes.new_zeros(1).squeeze()

        # L_attn
        if attn_maps is not None and gt_boxes_list is not None and gt_classes_list is not None:
            losses["attn"] = self.attn_loss(attn_maps, gt_boxes_list, gt_classes_list)
        else:
            losses["attn"] = pred_boxes.new_zeros(1).squeeze()

        # L_contrast
        if anchor is not None and positive is not None and negative is not None:
            losses["contrast"] = self.contrast_loss(anchor, positive, negative)
        else:
            losses["contrast"] = pred_boxes.new_zeros(1).squeeze()

        # L_physics
        if physics_implied is not None and predicted_class_logits is not None:
            losses["physics"] = self.physics_loss(physics_implied, predicted_class_logits)
        else:
            losses["physics"] = pred_boxes.new_zeros(1).squeeze()

        # Weighted total
        losses["total"] = (
            self.W_BOX * losses["box"]
            + self.W_CLS * losses["cls"]
            + self.W_DFL * losses["dfl"]
            + self.W_ATTN * losses["attn"]
            + self.W_CONTRAST * losses["contrast"]
            + self.W_PHYSICS * losses["physics"]
        )

        return losses
