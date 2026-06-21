"""Confidence-Gated Multi-Task Head — Stage 5 of XCarDamageNet.

Produces 6 outputs per detected damage region from multi-scale neck features (P3/P4/P5):
  1. Bounding box (anchor-free, x/y/w/h)
  2. Class probabilities (6 CarDD classes)
  3. Severity score (0-1)
  4. Damage cause (5 classes: impact/hail/vandalism/wear/environmental)
  5. Repair action (5 classes: PDR/panel_replacement/paint_refinish/glass_replacement/tire_replacement)
  6. Attention maps (B, 6, H/8, W/8) — one heatmap per class, for explainability
  7. Fraud score (0-1) — physics consistency check

Confidence gating: during inference, high-confidence predictions exit early
across 3 detection stages rather than running all 4. During training, all
stages run to accumulate exit losses.

~6M new parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict

NUM_CLASSES = 6   # dent, scratch, crack, glass_shatter, lamp_broken, tire_flat
NUM_CAUSES = 5    # impact, hail, vandalism, wear, environmental
NUM_REPAIRS = 5   # PDR, panel_replacement, paint_refinish, glass_replacement, tire_replacement

# Per-class severity weights (domain knowledge, not learnable)
CLASS_SEVERITY_WEIGHTS = [0.5, 0.3, 0.6, 0.9, 0.7, 1.0]  # dent,scratch,crack,glass,lamp,tire


class DetectionBranch(nn.Module):
    """Anchor-free detection head for a single feature scale.

    Predicts: objectness (1) + box coords (4) + class logits (NUM_CLASSES) per location.
    """

    def __init__(self, in_ch: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        mid_ch = max(in_ch // 2, 64)

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        )

        # Final prediction: objectness(1) + box(4) + classes(NUM_CLASSES)
        self.pred = nn.Conv2d(mid_ch, 1 + 4 + num_classes, 1)

        # Confidence gates at stages 1, 2, 3 (thresholds learnable)
        self.gate1 = nn.Linear(mid_ch, 1)
        self.gate2 = nn.Linear(mid_ch, 1)
        self.gate3 = nn.Linear(mid_ch, 1)

        # Learnable exit thresholds [0.90, 0.80, 0.70]
        self.thresh1 = nn.Parameter(torch.tensor(0.90))
        self.thresh2 = nn.Parameter(torch.tensor(0.80))
        self.thresh3 = nn.Parameter(torch.tensor(0.70))

    def _gate_confidence(
        self, features: torch.Tensor, gate: nn.Linear, threshold: nn.Parameter
    ) -> Tuple[torch.Tensor, bool]:
        """Compute confidence score for early exit decision.

        Args:
            features: (B, C, H, W)
            gate: linear layer mapping pooled features → confidence
            threshold: scalar threshold parameter

        Returns:
            gate_score: (B,) confidence score
            should_exit: whether max confidence exceeds threshold (inference only)
        """
        pooled = features.mean(dim=(-2, -1))  # (B, C)
        score = torch.sigmoid(gate(pooled)).squeeze(-1)  # (B,)
        should_exit = bool((score > threshold.clamp(0.5, 0.99)).all())
        return score, should_exit

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W) feature map from neck
            training: if False, apply early exit logic

        Returns:
            pred: (B, 1+4+NUM_CLASSES, H, W) detection predictions
            gate_scores: confidence gate scores at each stage [s1, s2, s3]
        """
        out = self.stage1(x)
        gate_scores = []

        s1, exit1 = self._gate_confidence(out, self.gate1, self.thresh1)
        gate_scores.append(s1)
        if not training and exit1:
            return self.pred(out), gate_scores

        out = self.stage2(out)
        s2, exit2 = self._gate_confidence(out, self.gate2, self.thresh2)
        gate_scores.append(s2)
        if not training and exit2:
            return self.pred(out), gate_scores

        out = self.stage3(out)
        s3, exit3 = self._gate_confidence(out, self.gate3, self.thresh3)
        gate_scores.append(s3)
        if not training and exit3:
            return self.pred(out), gate_scores

        out = self.stage4(out)
        gate_scores.append(torch.ones_like(s1))  # always exits at stage 4

        return self.pred(out), gate_scores


class SeverityHead(nn.Module):
    """Predicts damage severity (0-1) from ROI features + box geometry + class.

    Incorporates domain knowledge via class-specific severity weight multipliers
    (e.g. tire_flat always severe=1.0, scratch less severe=0.3).
    """

    def __init__(self, roi_feat_dim: int = 256) -> None:
        super().__init__()
        # roi_features (pooled to 1×1) + box_features (4) + class_probs (6) = roi_feat_dim + 10
        in_dim = roi_feat_dim + 10
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        # Register non-learnable class severity weights
        self.register_buffer(
            "class_weights",
            torch.tensor(CLASS_SEVERITY_WEIGHTS, dtype=torch.float32),
        )

    def forward(
        self,
        roi_features: torch.Tensor,
        box_xywh: torch.Tensor,
        class_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            roi_features: (B, roi_dim) — pooled ROI features
            box_xywh:     (B, 4) — [x, y, w, h] normalised to [0,1]
            class_probs:  (B, 6) — class probability vector

        Returns:
            severity: (B, 1) — severity score in [0, 1]
        """
        combined = torch.cat([roi_features, box_xywh, class_probs], dim=-1)
        base_severity = self.mlp(combined)  # (B, 1)

        # Scale by weighted class prediction
        class_weight = (class_probs * self.class_weights).sum(dim=-1, keepdim=True)
        severity = (base_severity * class_weight).clamp(0.0, 1.0)
        return severity


class CauseHead(nn.Module):
    """Predicts damage cause from ROI features (5-class softmax)."""

    def __init__(self, roi_feat_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(roi_feat_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, NUM_CAUSES),
        )

    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roi_features: (B, C) pooled ROI features
        Returns:
            cause_probs: (B, 5) — softmax probabilities
        """
        return F.softmax(self.mlp(roi_features), dim=-1)


class RepairHead(nn.Module):
    """Predicts repair action from ROI features + severity + cause (5-class softmax)."""

    def __init__(self, roi_feat_dim: int = 256) -> None:
        super().__init__()
        # roi_features + severity(1) + cause_probs(5) = C+6
        self.mlp = nn.Sequential(
            nn.Linear(roi_feat_dim + 6, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, NUM_REPAIRS),
        )

    def forward(
        self,
        roi_features: torch.Tensor,
        severity: torch.Tensor,
        cause_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            roi_features: (B, C)
            severity:     (B, 1)
            cause_probs:  (B, 5)
        Returns:
            repair_probs: (B, 5)
        """
        combined = torch.cat([roi_features, severity, cause_probs], dim=-1)
        return F.softmax(self.mlp(combined), dim=-1)


class AttentionMapHead(nn.Module):
    """Produces per-class damage heatmaps from P3 (finest resolution) features.

    Output: (B, NUM_CLASSES, H/8, W/8) — trained via AttentionSupervisionLoss
    to focus on actual damage regions (EU AI Act explainability requirement).
    """

    def __init__(self, p3_ch: int = 256, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(p3_ch, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
            nn.Sigmoid(),
        )

    def forward(self, p3: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p3: (B, 256, H/8, W/8) — finest scale features from neck
        Returns:
            attn_maps: (B, 6, H/8, W/8) — per-class heatmaps in [0,1]
        """
        return self.conv(p3)


class FraudHead(nn.Module):
    """Detects physics-inconsistent damage claims (fraud indicator).

    Compares physics-implied damage type (from physics tokens) with the
    predicted detection label. High disagreement → high fraud probability.
    """

    def __init__(self, physics_dim: int = 396) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(physics_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        # Also produces implied class logits for physics consistency loss
        self.implied_class = nn.Linear(64, NUM_CLASSES)

    def forward(
        self, physics_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            physics_tokens: (B, N, 396) — from physics encoder

        Returns:
            fraud_score:     (B, 1) — fraud probability in [0,1]
            implied_logits:  (B, NUM_CLASSES) — physics-implied damage class
        """
        # Aggregate physics tokens to image-level representation
        pooled = physics_tokens.mean(dim=1)  # (B, 396)

        hidden = self.mlp[0](pooled)  # Linear: (B, 64)
        hidden = self.mlp[1](hidden)  # SiLU
        implied_logits = self.implied_class(hidden)  # (B, NUM_CLASSES)

        fraud_score = self.mlp[2](hidden)  # Linear: (B, 1)
        fraud_score = self.mlp[3](fraud_score)  # Sigmoid

        return fraud_score, implied_logits


class ConfidenceGatedMultiTaskHead(nn.Module):
    """Complete multi-task detection head with confidence-gated early exit.

    Takes multi-scale features (P3, P4, P5) from DamageAwareNeck and
    physics tokens from PhysicsTokenEncoder, produces all 6 outputs.

    Detection is run at 3 scales (P3, P4, P5) as in anchor-free detectors.
    P3=fine (small damage), P4=medium (dents), P5=coarse (glass shatter, tire flat).

    Severity/Cause/Repair heads operate on P3-pooled ROI features.
    Attention maps from P3 (finest resolution).
    Fraud score from physics tokens.
    """

    def __init__(
        self,
        p3_ch: int = 256,
        p4_ch: int = 512,
        p5_ch: int = 512,
        physics_dim: int = 396,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        # Detection branches at each scale
        self.det_p3 = DetectionBranch(p3_ch, num_classes)
        self.det_p4 = DetectionBranch(p4_ch, num_classes)
        self.det_p5 = DetectionBranch(p5_ch, num_classes)

        # Downstream task heads (operate on P3 ROI features = 256-dim)
        self.severity = SeverityHead(roi_feat_dim=p3_ch)
        self.cause = CauseHead(roi_feat_dim=p3_ch)
        self.repair = RepairHead(roi_feat_dim=p3_ch)

        # Attention maps for explainability
        self.attn_map = AttentionMapHead(p3_ch, num_classes)

        # Fraud detection from physics tokens
        self.fraud = FraudHead(physics_dim)

        # Pool P3 features for non-detection heads (used when no ROI boxes available)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
        physics_tokens: torch.Tensor,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            p3:            (B, 256, H_p3, W_p3) — fine features from neck
            p4:            (B, 512, H_p4, W_p4) — medium features
            p5:            (B, 512, H_p5, W_p5) — coarse features
            physics_tokens:(B, N, 396) — from physics encoder
            training:      enables/disables early exit logic

        Returns dict with keys:
            "det_p3":      (B, 5+NUM_CLASSES, H_p3, W_p3)
            "det_p4":      (B, 5+NUM_CLASSES, H_p4, W_p4)
            "det_p5":      (B, 5+NUM_CLASSES, H_p5, W_p5)
            "severity":    (B, 1)
            "cause":       (B, NUM_CAUSES)
            "repair":      (B, NUM_REPAIRS)
            "attn_maps":   (B, NUM_CLASSES, H_p3, W_p3)
            "fraud_score": (B, 1)
            "fraud_implied":(B, NUM_CLASSES) — physics-implied class logits
            "gate_scores": list of confidence scores for loss computation
        """
        # Multi-scale detection
        det_p3, gs3 = self.det_p3(p3, training)  # (B, 5+C, H, W)
        det_p4, gs4 = self.det_p4(p4, training)
        det_p5, gs5 = self.det_p5(p5, training)

        # Pool P3 for image-level downstream task heads
        roi_feats = self.global_pool(p3).squeeze(-1).squeeze(-1)  # (B, 256)

        # Extract class probs from P3 detection for severity conditioning
        # avg over spatial → class logits → softmax
        p3_spatial_avg = det_p3.mean(dim=(-2, -1))  # (B, 5+C)
        class_logits = p3_spatial_avg[:, 5:]         # (B, NUM_CLASSES)
        class_probs = F.softmax(class_logits, dim=-1)

        # Dummy box features (global stats when no explicit ROI boxes)
        box_feats = torch.zeros(p3.shape[0], 4, device=p3.device)

        severity = self.severity(roi_feats, box_feats, class_probs)  # (B, 1)
        cause = self.cause(roi_feats)                                  # (B, 5)
        repair = self.repair(roi_feats, severity, cause)               # (B, 5)

        # Explainability heatmaps from P3
        attn_maps = self.attn_map(p3)  # (B, 6, H_p3, W_p3)

        # Fraud detection from physics tokens
        fraud_score, fraud_implied = self.fraud(physics_tokens)

        return {
            "det_p3": det_p3,
            "det_p4": det_p4,
            "det_p5": det_p5,
            "severity": severity,
            "cause": cause,
            "repair": repair,
            "attn_maps": attn_maps,
            "fraud_score": fraud_score,
            "fraud_implied": fraud_implied,
            "gate_scores": gs3 + gs4 + gs5,
        }
