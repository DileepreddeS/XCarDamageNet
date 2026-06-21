"""MAE Pre-trainer for PhysicsTokenEncoder self-supervised learning.

Pre-trains the PhysicsTokenEncoder on 50K unlabeled car images using
Masked Autoencoder (He et al., CVPR 2022):
    - Mask 75% of patches
    - Encoder processes visible patches
    - Decoder reconstructs masked patches
    - Physics heads predict surface properties

Hyperparameters from spec:
    epochs:   200-400
    lr:       1.5e-4 with cosine decay
    batch:    64
    aug:      random crop, hflip, color jitter
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional


class MAEDecoder(nn.Module):
    """Simple lightweight decoder for MAE reconstruction.

    Reconstructs pixel values from physics-augmented tokens.
    """

    def __init__(self, token_dim: int = 396, patch_size: int = 14) -> None:
        super().__init__()
        self.patch_size = patch_size
        out_dim = patch_size * patch_size * 3  # pixels per patch

        self.decoder = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, D) physics-augmented tokens
        Returns:
            pixels: (B, N, patch_size^2 * 3) reconstructed patches
        """
        return self.decoder(tokens)


class MAEPretrainer:
    """MAE pre-training loop for PhysicsTokenEncoder.

    Trains only the physics encoder + MAE decoder.
    DINOv2 backbone remains frozen during pre-training too.
    """

    def __init__(
        self,
        backbone,
        physics_encoder,
        train_loader: DataLoader,
        save_dir: str = "runs/pretrain",
        lr: float = 1.5e-4,
        epochs: int = 200,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.backbone = backbone.to(device)
        self.physics_encoder = physics_encoder.to(device)
        self.decoder = MAEDecoder().to(device)
        self.train_loader = train_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.epochs = epochs

        # Only train physics encoder + decoder
        params = list(physics_encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        self.recon_loss = nn.MSELoss()

    def train(self) -> None:
        print(f"MAE pre-training for {self.epochs} epochs on {self.device}")

        for epoch in range(self.epochs):
            self.physics_encoder.train()
            self.decoder.train()
            total_loss = 0.0

            for original, masked, mask in self.train_loader:
                original = original.to(self.device)
                masked = masked.to(self.device)
                mask = mask.to(self.device)

                with torch.no_grad():
                    tokens = self.backbone(masked)  # (B, N, 384)

                aug_tokens, _ = self.physics_encoder(tokens)  # (B, N, 396)
                reconstructed = self.decoder(aug_tokens)       # (B, N, patch^2*3)

                # Reconstruction loss on masked patches only
                # (loss = 0 on visible patches, matches MAE paper)
                loss = self.recon_loss(
                    reconstructed[mask], self._extract_targets(original, mask)
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            self.scheduler.step()
            avg = total_loss / max(1, len(self.train_loader))
            print(f"Pretrain epoch {epoch+1}/{self.epochs} | loss: {avg:.4f}")

            if (epoch + 1) % 50 == 0:
                torch.save(
                    self.physics_encoder.state_dict(),
                    self.save_dir / f"physics_epoch_{epoch+1:03d}.pt",
                )

    def _extract_targets(
        self, original: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract pixel targets for masked patches.

        Args:
            original: (B, 3, H, W)
            mask:     (B, N) bool — True = masked patch

        Returns:
            targets: (M, patch^2 * 3) where M = number of masked patches
        """
        B, C, H, W = original.shape
        p = self.decoder.patch_size
        grid = H // p

        # Unfold into patches: (B, C, grid, grid, p, p) → (B, N, C*p*p)
        patches = original.view(B, C, grid, p, grid, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(B, grid * grid, C * p * p)  # (B, N, C*p^2)

        # Select masked patches
        return patches[mask]  # (M, C*p^2)
