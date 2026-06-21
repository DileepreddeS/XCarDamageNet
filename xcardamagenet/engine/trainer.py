"""Training loop for XCarDamageNet fine-tuning on CarDD.

Hyperparameters from spec:
    optimizer:  AdamW
    lr:         1e-4
    weight_decay: 0.05
    schedule:   cosine decay to 1e-6
    warmup:     5 epochs
    epochs:     150
    patience:   30 (early stopping)
    batch_size: 8
    amp:        False  (NaN issues on V100 with mixed precision)
    save_period:10
"""

from __future__ import annotations

import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
from pathlib import Path

from ..models.xcardamagenet import XCarDamageNet
from ..losses.combined_loss import CombinedLoss
from ..utils.anchors import AnchorFreeTargetAssigner


class Trainer:
    """Fine-tuning trainer for XCarDamageNet on CarDD.

    Implements:
    - AdamW with cosine LR decay + warmup
    - Combined loss (box + cls + dfl + attn + contrast + physics)
    - Early stopping with patience
    - Checkpoint saving every save_period epochs
    - No AMP (amp=False, per spec — NaN issues observed on V100)
    """

    def __init__(
        self,
        model: XCarDamageNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str = "runs/train",
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        epochs: int = 150,
        warmup_epochs: int = 5,
        patience: int = 30,
        save_period: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.save_period = save_period

        # Optimise only trainable parameters (backbone frozen by default)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay
        )

        # Cosine LR schedule: decays from lr to 1e-6 over `epochs` steps
        min_lr = 1e-6
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=min_lr,
        )

        self.loss_fn = CombinedLoss()
        self.target_assigner = AnchorFreeTargetAssigner()

        self.best_map = 0.0
        self.epochs_without_improvement = 0

    def _warmup_lr(self, epoch: int, step: int, steps_per_epoch: int) -> None:
        """Linear warmup for the first `warmup_epochs` epochs."""
        if epoch >= self.warmup_epochs:
            return
        warmup_steps = self.warmup_epochs * steps_per_epoch
        current_step = epoch * steps_per_epoch + step
        lr_scale = current_step / max(1, warmup_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", 1e-4) * lr_scale

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one epoch of training."""
        self.model.train()
        total_losses = {"total": 0, "box": 0, "cls": 0}
        n_batches = len(self.train_loader)

        for step, (images, targets) in enumerate(self.train_loader):
            self._warmup_lr(epoch, step, n_batches)

            images = images.to(self.device)

            # Forward pass
            outputs = self.model(images, training=True)

            # Assign targets to predictions (simplified: use GT boxes directly)
            pred_boxes, gt_boxes, class_ids, pred_cls, gt_cls = (
                self.target_assigner.assign(outputs, targets, self.device)
            )

            if pred_boxes is None or pred_boxes.numel() == 0:
                continue

            # Compute losses
            losses = self.loss_fn(
                pred_boxes, gt_boxes, class_ids, pred_cls, gt_cls,
                attn_maps=outputs.get("attn_maps"),
                gt_boxes_list=[t["boxes"].to(self.device) for t in targets],
                gt_classes_list=[t["classes"].to(self.device) for t in targets],
                physics_implied=outputs.get("fraud_implied"),
                predicted_class_logits=pred_cls,
            )

            self.optimizer.zero_grad()
            losses["total"].backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            for k in ["total", "box", "cls"]:
                total_losses[k] += losses[k].item()

        if epoch >= self.warmup_epochs:
            self.scheduler.step()

        n = max(1, n_batches)
        return {k: v / n for k, v in total_losses.items()}

    def train(self) -> None:
        """Full training loop with early stopping and checkpointing."""
        print(f"Starting training for {self.epochs} epochs on {self.device}")
        print(f"  Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        for epoch in range(self.epochs):
            t0 = time.time()
            train_losses = self.train_one_epoch(epoch)
            elapsed = time.time() - t0

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1:3d}/{self.epochs} | "
                f"loss: {train_losses['total']:.4f} "
                f"(box:{train_losses['box']:.3f} cls:{train_losses['cls']:.3f}) | "
                f"lr: {lr:.2e} | {elapsed:.1f}s"
            )

            # Checkpoint every save_period epochs (CRITICAL for HPC job safety)
            if (epoch + 1) % self.save_period == 0:
                self._save_checkpoint(epoch, train_losses["total"])

            # Early stopping check (simplified — full validation with mAP
            # would use Validator class)
            if train_losses["total"] < self.best_map + 1e-4:
                self.best_map = train_losses["total"]
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, train_losses["total"], is_best=True)
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch+1} (patience={self.patience})")
                break

    def _save_checkpoint(
        self, epoch: int, loss: float, is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }
        suffix = "best.pt" if is_best else f"epoch_{epoch+1:03d}.pt"
        path = self.save_dir / suffix
        torch.save(state, path)
