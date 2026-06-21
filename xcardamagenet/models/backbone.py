"""DINOv2 ViT-S backbone wrapper for XCarDamageNet.

DINOv2 was self-supervised on 142M images, producing richer texture/surface
representations than COCO-supervised backbones — critical for damage detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from typing import Optional


class DINOv2Backbone(nn.Module):
    """Wraps DINOv2 ViT-S (22M params) for use as XCarDamageNet feature extractor.

    Outputs patch tokens (B, N, 384) suitable for downstream physics encoding.
    Optionally fine-tunes the last N transformer blocks while keeping earlier
    layers frozen — saves memory and prevents catastrophic forgetting.
    """

    MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
    EMBED_DIM = 384
    PATCH_SIZE = 14

    def __init__(
        self,
        pretrained: bool = True,
        unfreeze_last_n_blocks: int = 0,
        img_size: int = 518,
    ) -> None:
        """
        Args:
            pretrained: Load Meta's DINOv2 pre-trained weights.
            unfreeze_last_n_blocks: Number of transformer blocks from the end
                to leave trainable. 0 = fully frozen backbone.
            img_size: Input image spatial size. DINOv2 native = 518.
        """
        super().__init__()

        self.img_size = img_size
        self.embed_dim = self.EMBED_DIM
        self.patch_size = self.PATCH_SIZE

        # Load DINOv2 ViT-S — num_classes=0 removes the classification head
        self.backbone = timm.create_model(
            self.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )

        # Freeze all parameters by default
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Selectively unfreeze the last N transformer blocks
        if unfreeze_last_n_blocks > 0:
            blocks = list(self.backbone.blocks)
            for block in blocks[-unfreeze_last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            # Also unfreeze the final norm layer
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        n_patches_per_side = img_size // self.PATCH_SIZE
        self.n_patches = n_patches_per_side * n_patches_per_side
        self.grid_size = (n_patches_per_side, n_patches_per_side)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from input image.

        Args:
            x: Input image tensor, shape (B, 3, H, W).

        Returns:
            Patch token embeddings, shape (B, N_patches, 384).
            N_patches = (H/14) * (W/14).
        """
        # (B, 3, H, W) → (B, N+1, 384) where +1 is the [CLS] token
        features = self.backbone.forward_features(x)

        # Drop the [CLS] token, keep only patch tokens
        # DINOv2 ViT-S has class token at index 0
        patch_tokens = features[:, 1:, :]  # (B, N, 384)

        return patch_tokens

    def get_grid_size(self) -> tuple[int, int]:
        """Return (H_patches, W_patches) for spatial indexing of tokens."""
        return self.grid_size

    def trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        """Count all parameters."""
        return sum(p.numel() for p in self.parameters())
