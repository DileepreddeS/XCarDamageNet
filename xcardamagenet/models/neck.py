"""Texture Attention + Cross-Scale Neck — Stage 4 of XCarDamageNet.

Converts DINOv2 single-scale tokens into multi-scale spatial feature maps
(P3, P4, P5) then applies TextureAttention and CrossScaleAttention to detect
surface texture discontinuities and link damage evidence across spatial scales.

Damage = texture break. A hairline crack visible at P3 (fine) scale must be
linked to the impact zone visible at P5 (coarse) scale.

Architecture:
    1. Token→Map projection: (B, N, 396) → (B, 396, H/14, W/14)
    2. Multi-scale projection: stride-1/2/4 convs to create P3/P4/P5
    3. TextureAttention at each scale independently
    4. CrossScaleAttention between adjacent scales (P3↔P4, P4↔P5)
    5. DamageAwareNeck: top-down fusion P5→P4→P3

Output channel dims: P3=256, P4=512, P5=512.
~12M new parameters.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBnSilu(nn.Module):
    """Conv2d → BatchNorm2d → SiLU activation block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = -1,
    ) -> None:
        super().__init__()
        if padding < 0:
            padding = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TextureAttention(nn.Module):
    """Detect surface texture discontinuities via dual channel+spatial attention.

    Steps:
      1. Depthwise 3×3 conv extracts local texture patterns.
      2. Compute per-channel discontinuity = |texture - original|.
      3. Channel attention on discontinuity via squeeze-excite MLP.
      4. Spatial attention on discontinuity via avg+max pooling + conv.
      5. Output = original * channel_weight * spatial_weight + original (residual).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Depthwise conv to extract local texture without mixing channels
        self.texture_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )

        # Channel attention: squeeze-excite on discontinuity map
        reduced = max(channels // 8, 16)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, C, 1, 1)
            nn.Flatten(),                      # (B, C)
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

        # Spatial attention: avg+max pooling concat → 7×7 conv
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map (B, C, H, W)
        Returns:
            Texture-attended feature map (B, C, H, W)
        """
        texture = self.texture_conv(x)             # (B, C, H, W)
        disc = (texture - x).abs()                 # (B, C, H, W) discontinuity

        # Channel attention
        ch_w = self.channel_attn(disc)             # (B, C)
        ch_w = ch_w.unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)

        # Spatial attention using avg + max pooled channels
        avg_pool = disc.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool = disc.max(dim=1, keepdim=True).values  # (B, 1, H, W)
        sp_w = self.spatial_attn(torch.cat([avg_pool, max_pool], dim=1))  # (B, 1, H, W)

        return x * ch_w * sp_w + x  # residual


class CrossScaleAttention(nn.Module):
    """Attention between fine and coarse feature scales.

    Query from fine scale, Key/Value from coarse scale (upsampled to match).
    Allows damage evidence at one scale to inform another scale's features.
    """

    def __init__(self, fine_ch: int, coarse_ch: int) -> None:
        super().__init__()
        attn_dim = fine_ch // 4

        self.q_proj = nn.Conv2d(fine_ch, attn_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(coarse_ch, attn_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(coarse_ch, fine_ch, 1, bias=False)
        self.out_proj = nn.Conv2d(fine_ch, fine_ch, 1, bias=False)

        self.scale = attn_dim ** -0.5

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fine:   (B, C_fine,   H_f, W_f)
            coarse: (B, C_coarse, H_c, W_c) — upsampled to match fine spatially

        Returns:
            (B, C_fine, H_f, W_f) — fine features enriched with cross-scale context
        """
        B, C, H, W = fine.shape

        # Upsample coarse to match fine spatial resolution
        coarse_up = F.interpolate(coarse, size=(H, W), mode="bilinear", align_corners=False)

        q = self.q_proj(fine)       # (B, D, H, W)  where D = C//4
        k = self.k_proj(coarse_up)  # (B, D, H, W)
        v = self.v_proj(coarse_up)  # (B, C, H, W)

        D = q.shape[1]

        # Flatten spatial dims for attention
        q_flat = q.view(B, D, H * W).permute(0, 2, 1)    # (B, HW, D)
        k_flat = k.view(B, D, H * W)                       # (B, D, HW)
        v_flat = v.view(B, C, H * W).permute(0, 2, 1)     # (B, HW, C)

        # Scaled dot-product attention
        attn = torch.bmm(q_flat, k_flat) * self.scale      # (B, HW, HW)
        attn = F.softmax(attn, dim=-1)

        out_flat = torch.bmm(attn, v_flat)                  # (B, HW, C)
        out = out_flat.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)

        return fine + self.out_proj(out)  # residual


class TokenToMap(nn.Module):
    """Reshape flattened DINOv2 patch tokens back to 2D spatial feature maps.

    Tokens: (B, N, C) where N = H_p * W_p → Map: (B, C, H_p, W_p)
    Then project to create P3, P4, P5 at different spatial strides.
    """

    def __init__(
        self,
        token_dim: int = 396,
        p3_ch: int = 256,
        p4_ch: int = 512,
        p5_ch: int = 512,
    ) -> None:
        super().__init__()
        # stride=1: P3 resolution (H/14, W/14)
        self.to_p3 = ConvBnSilu(token_dim, p3_ch, kernel=1, stride=1, padding=0)
        # stride=2: P4 resolution (H/28, W/28)
        self.to_p4 = ConvBnSilu(token_dim, p4_ch, kernel=3, stride=2, padding=1)
        # stride=4: P5 resolution (H/56, W/56)
        self.to_p5 = nn.Sequential(
            ConvBnSilu(token_dim, p5_ch, kernel=3, stride=2, padding=1),
            ConvBnSilu(p5_ch, p5_ch, kernel=3, stride=2, padding=1),
        )

    def forward(
        self, tokens: torch.Tensor, grid_h: int, grid_w: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens:  (B, N, C) flattened patch tokens
            grid_h:  H_patches = H_img / patch_size
            grid_w:  W_patches = W_img / patch_size

        Returns:
            p3: (B, 256, grid_h,   grid_w)
            p4: (B, 512, grid_h/2, grid_w/2)
            p5: (B, 512, grid_h/4, grid_w/4)
        """
        B, N, C = tokens.shape
        # Reshape to spatial map
        feat_map = tokens.permute(0, 2, 1).view(B, C, grid_h, grid_w)  # (B, C, H_p, W_p)

        p3 = self.to_p3(feat_map)  # (B, 256, H_p,   W_p)
        p4 = self.to_p4(feat_map)  # (B, 512, H_p/2, W_p/2)
        p5 = self.to_p5(feat_map)  # (B, 512, H_p/4, W_p/4)

        return p3, p4, p5


class DamageAwareNeck(nn.Module):
    """Multi-scale feature fusion neck with texture awareness and cross-scale attention.

    Processing order (top-down):
        P5 → TextureAttn(P5) → align + upsample → fuse with P4
        P4 → TextureAttn(P4) + CrossScale(P4, P5) → align + upsample → fuse with P3
        P3 → TextureAttn(P3) + CrossScale(P3, P4)

    Output shapes preserved: P3'=(B,256,H,W), P4'=(B,512,H/2,W/2), P5'=(B,512,H/4,W/4)
    """

    def __init__(
        self,
        token_dim: int = 396,
        p3_ch: int = 256,
        p4_ch: int = 512,
        p5_ch: int = 512,
    ) -> None:
        super().__init__()

        self.token_to_map = TokenToMap(token_dim, p3_ch, p4_ch, p5_ch)

        # Texture attention at each scale
        self.tex_p5 = TextureAttention(p5_ch)
        self.tex_p4 = TextureAttention(p4_ch)
        self.tex_p3 = TextureAttention(p3_ch)

        # Cross-scale attention
        self.cross_p4_p5 = CrossScaleAttention(fine_ch=p4_ch, coarse_ch=p5_ch)
        self.cross_p3_p4 = CrossScaleAttention(fine_ch=p3_ch, coarse_ch=p4_ch)

        # Alignment convs for top-down fusion: 1×1 to match channels before addition
        self.align_p5_to_p4 = nn.Conv2d(p5_ch, p4_ch, 1, bias=False)
        self.align_p4_to_p3 = nn.Conv2d(p4_ch, p3_ch, 1, bias=False)

        # Output refinement convs after fusion
        self.out_p5 = ConvBnSilu(p5_ch, p5_ch)
        self.out_p4 = ConvBnSilu(p4_ch, p4_ch)
        self.out_p3 = ConvBnSilu(p3_ch, p3_ch)

    def forward(
        self,
        tokens: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: Physics-augmented patch tokens (B, N, 396)
            grid_h: Token grid height (H_img / patch_size)
            grid_w: Token grid width  (W_img / patch_size)

        Returns:
            p3_out: (B, 256, grid_h,   grid_w)   — fine scale
            p4_out: (B, 512, grid_h/2, grid_w/2) — medium scale
            p5_out: (B, 512, grid_h/4, grid_w/4) — coarse scale
        """
        # 1. Reshape tokens → multi-scale maps
        p3, p4, p5 = self.token_to_map(tokens, grid_h, grid_w)
        # p3: (B, 256, H, W), p4: (B, 512, H/2, W/2), p5: (B, 512, H/4, W/4)

        # 2. Texture attention at each scale
        p5 = self.tex_p5(p5)  # coarse: glass shatter, tire flat
        p4 = self.tex_p4(p4)  # medium: dents, lamp damage
        p3 = self.tex_p3(p3)  # fine:   cracks, thin scratches

        # 3. Top-down fusion: P5 → P4
        p5_aligned = self.align_p5_to_p4(p5)
        p5_up = F.interpolate(p5_aligned, size=p4.shape[2:], mode="bilinear", align_corners=False)
        p4 = p4 + p5_up
        p4 = self.cross_p4_p5(p4, p5)

        # 4. Top-down fusion: P4 → P3
        p4_aligned = self.align_p4_to_p3(p4)
        p4_up = F.interpolate(p4_aligned, size=p3.shape[2:], mode="bilinear", align_corners=False)
        p3 = p3 + p4_up
        p3 = self.cross_p3_p4(p3, p4)

        # 5. Final output convs
        p3_out = self.out_p3(p3)  # (B, 256, H,   W)
        p4_out = self.out_p4(p4)  # (B, 512, H/2, W/2)
        p5_out = self.out_p5(p5)  # (B, 512, H/4, W/4)

        return p3_out, p4_out, p5_out
