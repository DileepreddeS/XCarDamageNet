"""Physics Token Encoder — Stage 1 of XCarDamageNet.

Converts DINOv2 patch tokens (384-dim) into physics-informed tokens (396-dim)
by appending four parallel physics property predictions:
  - Surface Normal (3-dim unit vector)
  - Material Class (6-dim softmax: painted_metal, bare_metal, glass, rubber, plastic, other)
  - Reflectance Descriptor (2-dim after PCA reduction)
  - Surface Curvature (1-dim 0-1 magnitude)

Total: 384 + 3 + 6 + 2 + 1 = 396 dims.

The physics heads enable damage reasoning:
  "surface normal changed but material didn't" → dent
  "material changed, reflectance changed"      → scratch
  "curvature spike in local region"            → crease
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP2(nn.Module):
    """Two-layer MLP: Linear → GELU → Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SurfaceNormalHead(nn.Module):
    """Predicts a 3-dim unit normal vector per token.

    Output is L2-normalised so it always represents a valid direction vector.
    Self-supervised via photometric consistency during MAE pre-training.
    """

    def __init__(self, in_dim: int = 384) -> None:
        super().__init__()
        self.mlp = _MLP2(in_dim, 96, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 384)
        Returns:
            (B, N, 3) — unit surface normal vectors
        """
        normals = self.mlp(x)  # (B, N, 3)
        normals = F.normalize(normals, p=2, dim=-1)  # L2-normalize per token
        return normals


class MaterialHead(nn.Module):
    """Predicts material class probabilities per token.

    6 classes: painted_metal(0), bare_metal(1), glass(2), rubber(3), plastic(4), other(5).
    Trained with small supervised set (~2000 labeled patches) + self-supervised signals.
    """

    MATERIAL_CLASSES = ["painted_metal", "bare_metal", "glass", "rubber", "plastic", "other"]
    NUM_CLASSES = 6

    def __init__(self, in_dim: int = 384) -> None:
        super().__init__()
        self.mlp = _MLP2(in_dim, 96, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 384)
        Returns:
            (B, N, 6) — material class probabilities (softmax)
        """
        logits = self.mlp(x)  # (B, N, 6)
        return F.softmax(logits, dim=-1)


class ReflectanceHead(nn.Module):
    """Predicts reflectance descriptor per token, reduced to 2 dims.

    Full prediction is 4-dim; a learned linear projection reduces to 2 dims
    (simulating PCA reduction to keep total output manageable at 396 dims).
    Self-supervised: patches with similar normals should have similar reflectance.
    """

    def __init__(self, in_dim: int = 384) -> None:
        super().__init__()
        self.mlp = _MLP2(in_dim, 96, 4)
        # Learned 4→2 projection (replaces offline PCA for end-to-end training)
        self.reduce = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 384)
        Returns:
            (B, N, 2) — reflectance descriptor (PCA-reduced from 4)
        """
        desc = self.mlp(x)    # (B, N, 4)
        return self.reduce(desc)  # (B, N, 2)


class CurvatureHead(nn.Module):
    """Predicts surface curvature magnitude per token (0=flat, 1=high curvature).

    Self-supervised: smooth panels → low curvature, edges/creases → high curvature.
    Sigmoid ensures output stays in [0, 1].
    """

    def __init__(self, in_dim: int = 384) -> None:
        super().__init__()
        self.mlp = _MLP2(in_dim, 96, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 384)
        Returns:
            (B, N, 1) — curvature magnitude in [0, 1]
        """
        return torch.sigmoid(self.mlp(x))  # (B, N, 1)


class PhysicsTokenEncoder(nn.Module):
    """Four-head physics encoder that augments DINOv2 tokens with surface properties.

    Input:  (B, N, 384)  — raw DINOv2 patch tokens
    Output: (B, N, 396)  — physics-augmented tokens

    Concatenation order: [original_384 | normal_3 | material_6 | reflectance_2 | curvature_1]
    = 384 + 3 + 6 + 2 + 1 = 396

    ~1.2M parameters total (four 2-layer MLPs with hidden_dim=96).
    """

    OUTPUT_DIM = 396  # 384 + 3 + 6 + 2 + 1

    def __init__(self, in_dim: int = 384) -> None:
        """
        Args:
            in_dim: Input token dimension from DINOv2. Default 384.
        """
        super().__init__()
        self.in_dim = in_dim
        self.normal_head = SurfaceNormalHead(in_dim)
        self.material_head = MaterialHead(in_dim)
        self.reflectance_head = ReflectanceHead(in_dim)
        self.curvature_head = CurvatureHead(in_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Augment tokens with physics properties.

        Args:
            x: DINOv2 patch tokens, shape (B, N, 384).

        Returns:
            augmented: Physics-augmented tokens, shape (B, N, 396).
            physics_dict: Individual head outputs for loss computation:
                - "normal":      (B, N, 3)
                - "material":    (B, N, 6)
                - "reflectance": (B, N, 2)
                - "curvature":   (B, N, 1)
        """
        normal = self.normal_head(x)          # (B, N, 3)
        material = self.material_head(x)      # (B, N, 6)
        reflectance = self.reflectance_head(x)  # (B, N, 2)
        curvature = self.curvature_head(x)    # (B, N, 1)

        # Concatenate along feature dimension: 384+3+6+2+1 = 396
        augmented = torch.cat([x, normal, material, reflectance, curvature], dim=-1)
        # (B, N, 396)

        physics_dict = {
            "normal": normal,
            "material": material,
            "reflectance": reflectance,
            "curvature": curvature,
        }

        return augmented, physics_dict
