"""XCarDamageNet — Complete Hybrid CNN-Transformer Model.

Integrates all 5 processing stages into a unified end-to-end model:

    Stage 0: DINOv2 ViT-S Backbone      → (B, N, 384) patch tokens
    Stage 1: PhysicsTokenEncoder         → (B, N, 396) physics-augmented tokens
    Stage 2: AdaptiveInspectionAttention → (B, N, 396) + anomaly scores
    Stage 3: ContrastiveDamageModule     → (B, N, 396) + damage scores
    Stage 4: DamageAwareNeck             → (B,256,H,W), (B,512,H/2,W/2), (B,512,H/4,W/4)
    Stage 5: ConfidenceGatedMultiTaskHead→ 6 outputs per detection

Total: ~50M parameters (DINOv2 22M frozen + 28M novel components).

Usage:
    model = XCarDamageNet(img_size=518, pretrained=True)
    outputs = model(images)  # images: (B, 3, 518, 518)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .backbone import DINOv2Backbone
from .physics_encoder import PhysicsTokenEncoder
from .adaptive_attention import AdaptiveInspectionAttention
from .contrastive import ContrastiveDamageModule
from .neck import DamageAwareNeck
from .head import ConfidenceGatedMultiTaskHead


class XCarDamageNet(nn.Module):
    """Complete XCarDamageNet model for vehicle damage detection and assessment.

    A hybrid CNN-Transformer that detects surface anomalies rather than
    memorising damage appearances. Works on any vehicle, any damage type,
    any lighting condition — including damage types never seen in training.
    """

    def __init__(
        self,
        img_size: int = 518,
        pretrained_backbone: bool = True,
        unfreeze_backbone_blocks: int = 0,
        num_classes: int = 6,
    ) -> None:
        """
        Args:
            img_size: Input image size. DINOv2 native=518. Also supports 448, 392.
            pretrained_backbone: Load DINOv2 pre-trained weights from timm.
            unfreeze_backbone_blocks: Number of DINOv2 transformer blocks to
                fine-tune. 0 = fully frozen (recommended for small datasets).
                2 = last 2 blocks trainable.
            num_classes: Number of damage categories. 6 for CarDD.
        """
        super().__init__()

        self.img_size = img_size
        self.num_classes = num_classes

        # Stage 0: DINOv2 backbone
        self.backbone = DINOv2Backbone(
            pretrained=pretrained_backbone,
            unfreeze_last_n_blocks=unfreeze_backbone_blocks,
            img_size=img_size,
        )
        grid_h, grid_w = self.backbone.get_grid_size()
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Stage 1: Physics Token Encoder
        self.physics_encoder = PhysicsTokenEncoder(in_dim=384)

        # Stage 2: Adaptive Inspection Attention
        self.adaptive_attention = AdaptiveInspectionAttention(
            token_dim=396,
            coarse_attn_dim=96,
            grid_h=grid_h,
            grid_w=grid_w,
        )

        # Stage 3: Contrastive Damage Module
        self.contrastive = ContrastiveDamageModule(token_dim=396)

        # Stage 4: Damage-Aware Neck
        self.neck = DamageAwareNeck(token_dim=396)

        # Stage 5: Multi-Task Head
        self.head = ConfidenceGatedMultiTaskHead(
            p3_ch=256,
            p4_ch=512,
            p5_ch=512,
            physics_dim=396,
            num_classes=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        training: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass through all 5 stages.

        Args:
            x: Input image batch (B, 3, H, W).
            training: Override training mode for confidence gating.
                If None, uses self.training.

        Returns:
            Dict with keys from ConfidenceGatedMultiTaskHead plus:
                "anomaly_scores": (B, N) per-token anomaly probabilities
                "damage_scores":  (B, N) per-token damage scores
                "physics":        dict of physics head outputs
        """
        is_training = training if training is not None else self.training
        B = x.shape[0]

        # ── Stage 0: Extract DINOv2 patch tokens ──────────────────────────
        # (B, 3, H, W) → (B, N, 384)
        tokens = self.backbone(x)

        # ── Stage 1: Augment with physics properties ───────────────────────
        # (B, N, 384) → (B, N, 396) + physics_dict
        tokens, physics_dict = self.physics_encoder(tokens)

        # ── Stage 2: Two-pass adaptive inspection ──────────────────────────
        # (B, N, 396) → (B, N, 396) + (B, N) anomaly scores
        tokens, anomaly_scores = self.adaptive_attention(
            tokens, self.grid_h, self.grid_w
        )

        # Build suspicious mask from anomaly scores for contrastive module
        # Use 0.3 as inference threshold (matches CoarseScanEncoder default)
        suspicious_mask = (anomaly_scores > 0.3)  # (B, N)

        # ── Stage 3: Within-image contrastive damage detection ─────────────
        # (B, N, 396) → (B, N, 396) + (B, N) damage scores
        tokens, damage_scores = self.contrastive(tokens, suspicious_mask)

        # ── Stage 4: Multi-scale texture-aware neck ────────────────────────
        # (B, N, 396) → P3(B,256,H,W), P4(B,512,H/2,W/2), P5(B,512,H/4,W/4)
        p3, p4, p5 = self.neck(tokens, self.grid_h, self.grid_w)

        # ── Stage 5: Confidence-gated multi-task detection head ────────────
        outputs = self.head(p3, p4, p5, tokens, training=is_training)

        # Add auxiliary outputs used by loss functions and explainability
        outputs["anomaly_scores"] = anomaly_scores
        outputs["damage_scores"] = damage_scores
        outputs["physics"] = physics_dict

        return outputs

    def parameter_count(self) -> Dict[str, int]:
        """Return parameter counts broken down by module."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        def count_trainable(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "backbone_total": count(self.backbone),
            "backbone_trainable": count_trainable(self.backbone),
            "physics_encoder": count(self.physics_encoder),
            "adaptive_attention": count(self.adaptive_attention),
            "contrastive": count(self.contrastive),
            "neck": count(self.neck),
            "head": count(self.head),
            "total_trainable": count_trainable(self),
            "total": count(self),
        }
