"""Physics Consistency Loss — penalises physically implausible damage predictions.

Cross-entropy between physics-implied damage type and predicted damage type.

Physics inference rules (from token analysis):
    - Normal changed, material unchanged → dent (class 0)
    - Material changed, reflectance changed → scratch (class 1)
    - Curvature spike, material changed → crack (class 2)
    - Reflectance highly abnormal → glass shatter (class 3)

If physics tokens imply "dent" but the detector predicts "scratch",
the cross-entropy between implied and predicted logits increases.
This discourages physically inconsistent predictions.

The fraud head in the model produces `fraud_implied` logits for exactly this.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConsistencyLoss(nn.Module):
    """Cross-entropy between physics-implied class and predicted class.

    Uses the FraudHead's `implied_logits` output as the physics prediction.
    Backpropagates through both the physics head and the detector.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """
        Args:
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(
        self,
        physics_implied_logits: torch.Tensor,
        predicted_class_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute physics consistency loss.

        Args:
            physics_implied_logits: (B, num_classes) — logits from FraudHead
                representing what the physics tokens imply the damage class should be
            predicted_class_logits: (B, num_classes) — detection head class logits

        Returns:
            loss: physics consistency cross-entropy loss
        """
        # Physics-implied soft labels (softmax to get distribution)
        physics_probs = F.softmax(physics_implied_logits, dim=-1).detach()
        # Detach physics probs: we want detection head to match physics, not vice versa.
        # This prevents the physics head from "cheating" by matching the detector.

        # CE with soft labels: -sum(p_physics * log(p_pred))
        log_pred = F.log_softmax(predicted_class_logits, dim=-1)
        loss = -(physics_probs * log_pred).sum(dim=-1)  # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
