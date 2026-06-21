"""Adaptive Inspection Attention — Stage 2 of XCarDamageNet.

Mimics human inspector behaviour: scan the whole panel fast (Pass 1),
then zoom in on suspicious areas for detailed analysis (Pass 2).

Pass 1 — Coarse Scan (~3ms):
    2-layer Transformer encoder, 4 heads, compressed dim (396→96→396).
    Outputs anomaly_scores (B, N, 1) and a suspicious boolean mask.

Pass 2 — Fine Inspection (~2-8ms):
    4-layer Transformer encoder, 8 heads, full dim 396.
    Processes ONLY suspicious tokens + their 1-ring spatial neighbours.
    Non-suspicious tokens pass through unchanged (identity skip).

Saves 40-60% compute on clean images while increasing accuracy on damage.
~8M new parameters.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block: multi-head self-attention + FFN."""

    def __init__(self, dim: int, n_heads: int, ffn_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        ffn_dim = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CoarseScanEncoder(nn.Module):
    """Pass 1: fast 2-layer Transformer that assigns anomaly scores to all tokens.

    Projects 396→96 for attention to save compute, then back to 396.
    Anomaly scoring MLP produces per-token suspicious probability.
    """

    def __init__(self, token_dim: int = 396, attn_dim: int = 96) -> None:
        super().__init__()
        self.proj_down = nn.Linear(token_dim, attn_dim, bias=False)
        self.proj_up = nn.Linear(attn_dim, token_dim, bias=False)

        self.encoder = nn.Sequential(
            TransformerEncoderBlock(attn_dim, n_heads=4),
            TransformerEncoderBlock(attn_dim, n_heads=4),
        )

        # Anomaly scoring: per-token suspicious probability
        self.anomaly_mlp = nn.Sequential(
            nn.Linear(token_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Learnable threshold (init at logit corresponding to ~0.3 probability)
        init_logit = math.log(0.3 / (1 - 0.3))  # ≈ -0.847
        self.threshold_logit = nn.Parameter(torch.tensor(init_logit))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Physics-augmented tokens (B, N, 396)

        Returns:
            refined:       (B, N, 396) — coarse-refined tokens
            anomaly_scores:(B, N, 1)   — per-token suspicious probability [0,1]
            suspicious_mask:(B, N)     — bool, True = suspicious token
        """
        # Project down for efficient attention
        x_down = self.proj_down(x)   # (B, N, 96)
        x_down = self.encoder(x_down)  # (B, N, 96)
        delta = self.proj_up(x_down)   # (B, N, 396)
        refined = x + delta            # residual

        # Compute anomaly score from original + refined features
        anomaly_scores = self.anomaly_mlp(refined)  # (B, N, 1)

        # Learnable threshold via sigmoid to keep in (0, 1)
        threshold = torch.sigmoid(self.threshold_logit)  # scalar
        suspicious_mask = (anomaly_scores.squeeze(-1) > threshold)  # (B, N)

        return refined, anomaly_scores, suspicious_mask


class FineInspectionEncoder(nn.Module):
    """Pass 2: deep 4-layer Transformer that processes suspicious tokens + neighbours.

    Only operates on tokens where suspicious_mask=True plus their 1-ring
    spatial neighbours. Unflagged tokens pass through unchanged.
    ~6.5M parameters (most of the ~8M budget).
    """

    def __init__(self, token_dim: int = 396) -> None:
        super().__init__()
        # 396 is not divisible by 8, use 12 heads (396/12=33 per head)
        # Closest divisor of 396 to the spec's target of 8 heads
        n_heads = 12
        self.encoder = nn.Sequential(
            TransformerEncoderBlock(token_dim, n_heads=n_heads),
            TransformerEncoderBlock(token_dim, n_heads=n_heads),
            TransformerEncoderBlock(token_dim, n_heads=n_heads),
            TransformerEncoderBlock(token_dim, n_heads=n_heads),
        )

    def _get_neighbor_indices(
        self,
        suspicious_mask: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """Expand suspicious mask to include 1-ring spatial neighbours.

        Args:
            suspicious_mask: (N,) boolean mask for a single image
            grid_h, grid_w: spatial grid dimensions (N = grid_h * grid_w)

        Returns:
            expanded_mask: (N,) boolean mask including neighbours
        """
        mask_2d = suspicious_mask.view(grid_h, grid_w)  # (H, W)
        # Dilate with max-pool: adds 1-ring neighbours
        mask_float = mask_2d.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        dilated = F.max_pool2d(mask_float, kernel_size=3, stride=1, padding=1)
        return (dilated.squeeze() > 0.5)  # (H, W) → .view(N) later

    def forward(
        self,
        tokens: torch.Tensor,
        suspicious_mask: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens:          (B, N, 396)
            suspicious_mask: (B, N) bool
            grid_h, grid_w:  spatial grid dimensions

        Returns:
            (B, N, 396) — suspicious tokens updated, others unchanged
        """
        B, N, D = tokens.shape
        output = tokens.clone()

        for b in range(B):
            expanded = self._get_neighbor_indices(
                suspicious_mask[b], grid_h, grid_w
            ).view(N)  # (N,)

            if expanded.sum() == 0:
                continue  # No suspicious tokens in this image

            # Extract subset of tokens for fine processing
            subset = tokens[b][expanded].unsqueeze(0)  # (1, n_subset, D)
            refined = self.encoder(subset).squeeze(0)  # (n_subset, D)
            output[b][expanded] = refined

        return output


class AdaptiveInspectionAttention(nn.Module):
    """Two-pass adaptive attention: fast coarse scan → targeted fine inspection.

    Pass 1 identifies suspicious regions in ~3ms.
    Pass 2 applies deep processing only to suspicious regions + neighbours in ~2-8ms.
    Clean images (few suspicious tokens) are processed cheaply.

    Input:  (B, N, 396) physics tokens
    Output: (B, N, 396) refined tokens + (B, N) anomaly scores
    """

    def __init__(
        self,
        token_dim: int = 396,
        coarse_attn_dim: int = 96,
        grid_h: int = 37,
        grid_w: int = 37,
    ) -> None:
        """
        Args:
            token_dim: Token feature dimension (396 from physics encoder).
            coarse_attn_dim: Compressed dim for Pass 1 attention (96).
            grid_h, grid_w: Spatial grid size. Store for neighbour computation.
        """
        super().__init__()
        self.token_dim = token_dim
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.coarse = CoarseScanEncoder(token_dim, coarse_attn_dim)
        self.fine = FineInspectionEncoder(token_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        grid_h: Optional[int] = None,
        grid_w: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens:  (B, N, 396)
            grid_h:  Override stored grid height (useful when image size varies).
            grid_w:  Override stored grid width.

        Returns:
            refined:       (B, N, 396) — tokens after both inspection passes
            anomaly_scores:(B, N)      — per-token anomaly probability [0,1]
        """
        H = grid_h if grid_h is not None else self.grid_h
        W = grid_w if grid_w is not None else self.grid_w

        # Pass 1: Coarse scan — all tokens, cheap
        coarse_refined, anomaly_scores, suspicious_mask = self.coarse(tokens)
        # coarse_refined: (B, N, 396), anomaly_scores: (B, N, 1), suspicious_mask: (B, N)

        # Pass 2: Fine inspection — suspicious tokens + neighbours only
        fine_refined = self.fine(coarse_refined, suspicious_mask, H, W)
        # fine_refined: (B, N, 396)

        return fine_refined, anomaly_scores.squeeze(-1)  # (B, N, 396), (B, N)
