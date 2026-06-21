"""Class-Balanced BCE Loss using effective number of samples.

Addresses the 11.4× class imbalance in CarDD (scratch=2560, tire_flat=225).

Formula (Cui et al., CVPR 2019):
    effective_n_i = (1 - beta^n_i) / (1 - beta)
    weight_i = 1 / effective_n_i
    weights normalised to sum = num_classes

CarDD effective weights (beta=0.9999):
    dent=0.32, scratch=0.23, crack=0.84, glass=1.14, lamp=1.10, tire=2.37

These are registered as non-learnable buffers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# CarDD class sample counts: [dent, scratch, crack, glass_shatter, lamp_broken, tire_flat]
_CARDD_COUNTS = [1847, 2560, 659, 424, 429, 225]

# Pre-computed effective weights for beta=0.9999
_EFFECTIVE_WEIGHTS = [0.32, 0.23, 0.84, 1.14, 1.10, 2.37]


def compute_effective_weights(
    class_counts: list[int], beta: float = 0.9999
) -> torch.Tensor:
    """Compute class-balanced weights from effective number of samples.

    Args:
        class_counts: Number of training samples per class.
        beta: Hyperparameter controlling effective number. 0.9999 typical.

    Returns:
        weights: Normalised per-class weights summing to num_classes.
    """
    n = len(class_counts)
    eff_num = torch.tensor(
        [(1.0 - beta ** c) / (1.0 - beta) for c in class_counts], dtype=torch.float32
    )
    weights = 1.0 / eff_num
    weights = weights / weights.sum() * n  # normalise to sum = num_classes
    return weights


class ClassBalancedBCELoss(nn.Module):
    """Weighted binary cross-entropy with class-balanced sample weights.

    Suitable for multi-label detection where each anchor independently predicts
    each class. For hard positive/negative assignment, use with class_ids.
    """

    def __init__(
        self,
        class_counts: Optional[list[int]] = None,
        beta: float = 0.9999,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            class_counts: Per-class training sample counts.
                Defaults to CarDD counts.
            beta: Effective number hyperparameter.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

        if class_counts is None:
            class_counts = _CARDD_COUNTS

        weights = compute_effective_weights(class_counts, beta)
        self.register_buffer("class_weights", weights)  # (num_classes,)

    def forward(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute class-balanced BCE loss.

        Args:
            pred_logits: (..., num_classes) — raw logits before sigmoid
            targets:     (..., num_classes) — binary targets in [0, 1]

        Returns:
            loss: scalar (mean/sum) or (..., num_classes) tensor (none)
        """
        # Per-class BCE: (*, num_classes)
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, targets, reduction="none"
        )

        # Apply class-balanced weights
        # Broadcast class_weights to match bce shape
        weighted = bce * self.class_weights  # broadcasting over last dim

        if self.reduction == "mean":
            return weighted.mean()
        elif self.reduction == "sum":
            return weighted.sum()
        return weighted

    def sample_weights(self, class_ids: torch.Tensor) -> torch.Tensor:
        """Return per-sample class-balanced weights for use in other losses.

        Args:
            class_ids: (N,) integer class indices

        Returns:
            weights: (N,) per-sample weights
        """
        return self.class_weights[class_ids.clamp(0, len(_CARDD_COUNTS) - 1)]
