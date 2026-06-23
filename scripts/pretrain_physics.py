"""MAE pre-training for XCarDamageNet PhysicsTokenEncoder.

Trains the physics encoder heads (normal, material, reflectance, curvature)
on unlabeled car images using Masked Autoencoder (He et al., CVPR 2022).
DINOv2 backbone remains fully frozen throughout — we are teaching the physics
heads to interpret what DINOv2 already sees, not retraining the backbone.

Mask ratio: 75% of patches masked per step (standard MAE protocol).
Loss: MSE reconstruction of masked patch pixels via lightweight decoder.

Resume: pass --resume path/to/checkpoint.pt to continue exactly where stopped.
The checkpoint stores epoch, model, optimizer, and scheduler states.

Time limit: pass --time_limit 8 to stop cleanly after 8 hours (for HPC jobs
that have walltime limits). Training resumes from the last saved checkpoint.

Usage:
    python scripts/pretrain_physics.py \\
        --data_dir /path/to/car/images \\
        --output_dir ./runs/pretrain_001 \\
        --checkpoint_dir ./checkpoints/pretrain_001 \\
        --epochs 200 --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from xcardamagenet.models.backbone import DINOv2Backbone
from xcardamagenet.models.physics_encoder import PhysicsTokenEncoder
from xcardamagenet.data.pretrain_dataset import UnlabeledCarDataset


# ─────────────────────────────────────────────────────────────
# Decoder (lightweight — only trained during pre-training)
# ─────────────────────────────────────────────────────────────

class _MAEDecoder(nn.Module):
    """Two-layer MLP decoder: reconstructs patch pixels from physics tokens."""

    PATCH_SIZE = 14  # DINOv2 patch size

    def __init__(self, token_dim: int = 396) -> None:
        super().__init__()
        out_dim = self.PATCH_SIZE * self.PATCH_SIZE * 3
        self.net = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)  # (B, N, patch^2*3)


def _extract_patch_targets(
    images: torch.Tensor, mask: torch.Tensor, patch_size: int = 14
) -> torch.Tensor:
    """Extract pixel values for masked patches as regression targets.

    Args:
        images: (B, 3, H, W) normalised images
        mask:   (B, N) bool — True = masked patch (N = grid^2)

    Returns:
        targets: (M, 3*patch^2) where M = total masked patches across batch
    """
    B, C, H, W = images.shape
    p = patch_size
    g = H // p  # grid size per side

    # Reshape into patches: (B, grid, grid, C, p, p) → (B, N, C*p*p)
    x = images.view(B, C, g, p, g, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(B, g * g, C * p * p)  # (B, N, C*p^2)

    return x[mask]  # (M, C*p^2)


# ─────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────

def _save_checkpoint(
    ckpt_dir: Path,
    epoch: int,
    physics_encoder: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_loss: float,
    loss_history: list[float],
    label: str = "latest",
) -> None:
    state = {
        "epoch": epoch,
        "physics_encoder": physics_encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_loss": best_loss,
        "loss_history": loss_history,
    }
    path = ckpt_dir / f"pretrain_{label}.pt"
    torch.save(state, path)
    print(f"  Checkpoint saved → {path}")


def _load_checkpoint(
    path: str,
    physics_encoder: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
) -> tuple[int, float, list[float]]:
    state = torch.load(path, map_location=device)
    physics_encoder.load_state_dict(state["physics_encoder"])
    decoder.load_state_dict(state["decoder"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    epoch = state["epoch"]
    best_loss = state.get("best_loss", float("inf"))
    loss_history = state.get("loss_history", [])
    print(f"  Resumed from epoch {epoch + 1} (best_loss={best_loss:.6f})")
    return epoch, best_loss, loss_history


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Data:   {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Ckpts:  {args.checkpoint_dir}")

    out_dir = Path(args.output_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset & DataLoader ──────────────────────────────────
    dataset = UnlabeledCarDataset(
        image_dir=args.data_dir,
        img_size=args.image_size,
        mask_ratio=args.mask_ratio,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    print(f"Dataset: {len(dataset):,} images  |  {len(loader):,} batches/epoch")

    # ── Model components ──────────────────────────────────────
    print("Loading DINOv2 backbone (frozen)…")
    backbone = DINOv2Backbone(
        pretrained=True,
        unfreeze_last_n_blocks=0,
        img_size=args.image_size,
    ).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    physics_encoder = PhysicsTokenEncoder(in_dim=384).to(device)
    decoder = _MAEDecoder(token_dim=396).to(device)

    trainable_params = list(physics_encoder.parameters()) + list(decoder.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable:,}  (physics_encoder + MAE decoder)")

    # ── Optimiser & scheduler ──────────────────────────────────
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.05)
    # Cosine schedule (total = epochs - warmup, then held at min_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=1e-6,
    )
    recon_loss_fn = nn.MSELoss()

    # ── Resume from checkpoint ─────────────────────────────────
    start_epoch = 0
    best_loss = float("inf")
    loss_history: list[float] = []

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"WARNING: --resume path not found: {resume_path}  — starting fresh")
        else:
            start_epoch, best_loss, loss_history = _load_checkpoint(
                str(resume_path), physics_encoder, decoder, optimizer, scheduler, device
            )
            start_epoch += 1  # resume from next epoch

    # ── Deadline tracking for HPC time limits ─────────────────
    deadline: float | None = None
    if args.time_limit is not None:
        deadline = time.time() + args.time_limit * 3600
        print(f"Time limit: {args.time_limit}h  (stops gracefully before cutoff)")

    # ── Training loop ──────────────────────────────────────────
    print(f"\nPre-training for {args.epochs} epochs (starting from {start_epoch + 1})")
    print("-" * 70)

    complete_flag = out_dir / "pretrain_complete.flag"

    for epoch in range(start_epoch, args.epochs):
        # Check time limit before starting epoch
        if deadline is not None and time.time() >= deadline:
            print(f"\nTime limit reached at epoch {epoch + 1}. Saving checkpoint.")
            _save_checkpoint(ckpt_dir, epoch - 1, physics_encoder, decoder,
                             optimizer, scheduler, best_loss, loss_history)
            break

        physics_encoder.train()
        decoder.train()

        epoch_loss = 0.0
        t0 = time.time()

        for step, (original, masked, mask) in enumerate(loader):
            # Apply LR warmup (linear ramp over first warmup_epochs)
            if epoch < args.warmup_epochs:
                total_warmup_steps = args.warmup_epochs * len(loader)
                current_step = epoch * len(loader) + step
                lr_scale = (current_step + 1) / max(1, total_warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * lr_scale

            original = original.to(device)
            masked = masked.to(device)
            mask = mask.to(device)  # (B, N) bool

            # Backbone forward (no_grad — frozen)
            with torch.no_grad():
                tokens_raw = backbone(masked)  # (B, N, 384)

            # Physics encoder + MAE decoder
            tokens_aug, _ = physics_encoder(tokens_raw)   # (B, N, 396)
            reconstructed = decoder(tokens_aug)            # (B, N, 3*patch^2)

            # Loss only on masked patches (MAE protocol)
            targets = _extract_patch_targets(original, mask, _MAEDecoder.PATCH_SIZE)
            if targets.numel() == 0:
                continue

            pred_masked = reconstructed[mask]  # (M, 3*patch^2)
            loss = recon_loss_fn(pred_masked, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # Step scheduler after warmup
        if epoch >= args.warmup_epochs:
            scheduler.step()

        avg_loss = epoch_loss / max(1, len(loader))
        loss_history.append(avg_loss)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        print(
            f"Epoch {epoch+1:4d}/{args.epochs}  "
            f"loss={avg_loss:.6f}{'*' if is_best else ' '}  "
            f"lr={lr_now:.2e}  {elapsed:.0f}s"
        )

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0 or is_best:
            label = "best" if is_best else "latest"
            _save_checkpoint(ckpt_dir, epoch, physics_encoder, decoder,
                             optimizer, scheduler, best_loss, loss_history, label)
            # Always keep a "latest" copy too
            if is_best:
                _save_checkpoint(ckpt_dir, epoch, physics_encoder, decoder,
                                 optimizer, scheduler, best_loss, loss_history, "latest")

    else:
        # Loop completed without break → training finished
        _save_checkpoint(ckpt_dir, args.epochs - 1, physics_encoder, decoder,
                         optimizer, scheduler, best_loss, loss_history, "final")
        complete_flag.write_text(
            f"Pre-training completed: {args.epochs} epochs, best_loss={best_loss:.6f}\n"
        )
        print(f"\nPre-training complete. Flag written to {complete_flag}")

    # ── Save loss curve ────────────────────────────────────────
    _save_loss_curve(loss_history, out_dir / "pretrain_loss.png")

    # Save loss history as JSON for later analysis
    (out_dir / "pretrain_loss.json").write_text(
        json.dumps({"loss_history": loss_history, "best_loss": best_loss}, indent=2)
    )

    print(f"\nBest loss: {best_loss:.6f}")
    print(f"Outputs saved to: {out_dir}")


def _save_loss_curve(loss_history: list[float], path: Path) -> None:
    """Save loss curve as PNG. Skips gracefully if matplotlib not available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        epochs = list(range(1, len(loss_history) + 1))
        ax.plot(epochs, loss_history, linewidth=1.5, color="#2196F3", label="MAE recon loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Physics Encoder MAE Pre-training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        print(f"  Loss curve → {path}")
    except Exception as e:
        print(f"  (Could not save loss curve: {e})")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MAE pre-training for XCarDamageNet PhysicsTokenEncoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--data_dir",       required=True,
                   help="Directory containing unlabeled car images (recursively searched)")
    p.add_argument("--output_dir",     required=True,
                   help="Directory for run outputs (loss curves, logs, flags)")
    p.add_argument("--checkpoint_dir", required=True,
                   help="Directory for model checkpoints (.pt files)")

    # Optional
    p.add_argument("--resume",         default=None,
                   help="Path to checkpoint .pt file to resume from")

    # Hyperparameters (spec defaults)
    p.add_argument("--epochs",         type=int,   default=200)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1.5e-4)
    p.add_argument("--warmup_epochs",  type=int,   default=10)
    p.add_argument("--mask_ratio",     type=float, default=0.75,
                   help="Fraction of patches masked per image (MAE protocol)")
    p.add_argument("--image_size",     type=int,   default=518,
                   help="Input image size (DINOv2 native=518)")
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--save_every",     type=int,   default=20,
                   help="Save checkpoint every N epochs")
    p.add_argument("--time_limit",     type=float, default=None,
                   help="Stop after N hours (HPC walltime safety). None = no limit.")
    p.add_argument("--amp",            action="store_true", default=False,
                   help="Enable AMP (disabled by default — NaN risk on V100)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
