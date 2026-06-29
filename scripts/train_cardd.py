"""Fine-tune XCarDamageNet on CarDD dataset.

Trains the full model (all novel modules) while keeping DINOv2 backbone frozen
for the first --freeze_backbone_epochs, then optionally unfreezes the last 2
transformer blocks for the remainder.

Loss formula (from CLAUDE.md):
    L = 7.5*L_box + 0.5*L_cls + 1.5*L_dfl + 0.10*L_attn + 0.05*L_contrast + 0.02*L_physics

All 6 loss weights are overridable via CLI.

Milestone benchmarks (printed when exceeded):
    0.496 — DCN+ baseline
    0.700 — XCarDamage v1 initial
    0.730 — YOLOv9-CS
    0.739 — XCarDamage v1 with TTA

Resume: --resume restores model + optimizer + scheduler + epoch + best_map.
Time limit: --time_limit 20 stops cleanly before HPC walltime kills the job.

Usage:
    python scripts/train_cardd.py \\
        --data_dir /path/to/cardd \\
        --output_dir ./runs/train_001 \\
        --checkpoint_dir ./checkpoints/train_001 \\
        --physics_checkpoint ./checkpoints/pretrain_001/pretrain_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from xcardamagenet.models.xcardamagenet import XCarDamageNet
from xcardamagenet.models.physics_encoder import PhysicsTokenEncoder
from xcardamagenet.losses.combined_loss import CombinedLoss
from xcardamagenet.data.cardd_dataset import CarDDDataset, CLASS_NAMES
from xcardamagenet.data.augmentation import CarDDAugmentation, CopyPasteAugmentation
from xcardamagenet.utils.anchors import AnchorFreeTargetAssigner
from xcardamagenet.utils.metrics import compute_map

# Known benchmark mAP@0.5 values — print alert when exceeded
MILESTONES = [
    (0.496, "DCN+ baseline"),
    (0.700, "XCarDamage v1 initial"),
    (0.730, "YOLOv9-CS"),
    (0.739, "XCarDamage v1 + TTA"),
]


# ─────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────

def _save_checkpoint(
    ckpt_dir: Path,
    label: str,
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler,
    best_map: float,
    loss_history: list,
    map_history: list,
) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_map": best_map,
        "loss_history": loss_history,
        "map_history": map_history,
    }
    path = ckpt_dir / f"{label}.pt"
    torch.save(state, path)
    print(f"  Saved → {path}")


def _load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    device: str,
) -> tuple[int, float, list, list]:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    epoch = state["epoch"]
    best_map = state.get("best_map", 0.0)
    loss_history = state.get("loss_history", [])
    map_history = state.get("map_history", [])
    print(f"  Resumed from epoch {epoch + 1}  best_mAP={best_map:.4f}")
    return epoch, best_map, loss_history, map_history


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    conf_threshold: float = 0.25,
) -> dict[str, float]:
    """Run validation and return mAP metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    for images, targets in val_loader:
        images = images.to(device)
        outputs = model(images, training=False)

        det = outputs["det_p3"]  # (B, 5+C, H, W) finest scale
        B = det.shape[0]

        for b in range(B):
            pred_map = det[b]
            obj = torch.sigmoid(pred_map[0]).view(-1)           # (HW,)
            cls_logits = pred_map[5:].view(pred_map.shape[0] - 5, -1).T  # (HW, C)
            cls_probs = torch.softmax(cls_logits, dim=-1)
            scores, classes = cls_probs.max(dim=-1)
            scores = scores * obj

            keep = scores > conf_threshold
            # Dummy boxes (no proper coordinate decoding yet — zero boxes)
            n_keep = keep.sum().item()
            all_preds.append({
                "boxes":   torch.zeros(n_keep, 4),
                "scores":  scores[keep].cpu(),
                "classes": classes[keep].cpu(),
            })
            all_targets.append({
                "boxes":   targets[b]["boxes"],
                "classes": targets[b]["classes"],
            })

    metrics_50 = compute_map(all_preds, all_targets, iou_threshold=0.5)

    # mAP@0.5:0.95 (average over 10 thresholds)
    map_95_scores = []
    for iou_t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        m = compute_map(all_preds, all_targets, iou_threshold=iou_t)
        map_95_scores.append(m["mAP"])
    metrics_50["mAP@0.5:0.95"] = float(sum(map_95_scores) / len(map_95_scores))

    model.train()
    return metrics_50


# ─────────────────────────────────────────────────────────────
# Backbone freeze/unfreeze
# ─────────────────────────────────────────────────────────────

def _set_backbone_frozen(model: XCarDamageNet, frozen: bool) -> None:
    """Freeze or unfreeze backbone. When unfreezing, only last 2 blocks."""
    if frozen:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    else:
        # Unfreeze last 2 transformer blocks + final norm
        blocks = list(model.backbone.backbone.blocks)
        for block in blocks[-2:]:
            for p in block.parameters():
                p.requires_grad_(True)
        for p in model.backbone.backbone.norm.parameters():
            p.requires_grad_(True)
        n = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        print(f"  Backbone last-2-blocks unfrozen ({n:,} trainable params added)")


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:    {device}")
    print(f"Data:      {args.data_dir}")
    print(f"Output:    {args.output_dir}")
    print(f"Ckpts:     {args.checkpoint_dir}")

    out_dir = Path(args.output_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load YAML config (override defaults if provided) ──────
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        for k, v in cfg.items():
            if not getattr(args, k, None):
                setattr(args, k, v)

    # ── Datasets ──────────────────────────────────────────────
    aug = CarDDAugmentation(
        img_size=args.image_size,
        mosaic_prob=args.mosaic,
        mixup_prob=args.mixup,
        copy_paste_prob=args.copy_paste,
    )
    copy_paste = CopyPasteAugmentation(prob=args.copy_paste)

    train_ds = CarDDDataset(args.data_dir, split="train",
                            img_size=args.image_size, transforms=aug)
    val_ds = CarDDDataset(args.data_dir, split="val",
                          img_size=args.image_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=CarDDDataset.collate_fn,
        pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=CarDDDataset.collate_fn,
    )

    print(f"Train: {len(train_ds):,} images  |  Val: {len(val_ds):,} images")

    # ── Model ──────────────────────────────────────────────────
    print("Building XCarDamageNet…")
    model = XCarDamageNet(
        img_size=args.image_size,
        pretrained_backbone=True,
        unfreeze_backbone_blocks=0,  # start frozen, unfreeze later per epoch schedule
        num_classes=6,
    ).to(device)

    # Load pre-trained physics encoder weights if provided
    if args.physics_checkpoint:
        pc_path = Path(args.physics_checkpoint)
        if not pc_path.exists():
            print(f"WARNING: --physics_checkpoint not found: {pc_path}")
        else:
            ckpt = torch.load(str(pc_path), map_location=device)
            # Checkpoint may store full pretrain state or just physics_encoder weights
            if "physics_encoder" in ckpt:
                model.physics_encoder.load_state_dict(ckpt["physics_encoder"])
            else:
                model.physics_encoder.load_state_dict(ckpt)
            print(f"  Physics encoder loaded from {pc_path}")

    counts = model.parameter_count()
    print(f"  Total params: {counts['total']:,}  |  Trainable: {counts['total_trainable']:,}")

    # ── Loss ───────────────────────────────────────────────────
    loss_fn = CombinedLoss()
    loss_fn.W_BOX     = args.loss_box
    loss_fn.W_CLS     = args.loss_cls
    loss_fn.W_DFL     = args.loss_dfl
    loss_fn.W_ATTN    = args.loss_attn
    loss_fn.W_CONTRAST = args.loss_contrast
    loss_fn.W_PHYSICS  = args.loss_physics

    print(
        f"  Loss weights: box={args.loss_box}  cls={args.loss_cls}  "
        f"dfl={args.loss_dfl}  attn={args.loss_attn}  "
        f"contrast={args.loss_contrast}  physics={args.loss_physics}"
    )

    target_assigner = AnchorFreeTargetAssigner()

    # ── Optimiser (trainable params only) ─────────────────────
    def _get_trainable_params():
        return [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        _get_trainable_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=1e-6,
    )

    # ── Resume ────────────────────────────────────────────────
    start_epoch = 0
    best_map = 0.0
    loss_history: list[dict] = []
    map_history: list[float] = []
    milestones_beaten: set[float] = set()

    if args.resume:
        rp = Path(args.resume)
        if rp.exists():
            start_epoch, best_map, loss_history, map_history = _load_checkpoint(
                str(rp), model, optimizer, scheduler, device
            )
            start_epoch += 1
            # Mark already-beaten milestones so we don't re-alert
            for threshold, _ in MILESTONES:
                if best_map > threshold:
                    milestones_beaten.add(threshold)
        else:
            print(f"WARNING: --resume not found: {rp}  — starting fresh")

    # ── Time limit ────────────────────────────────────────────
    deadline: float | None = None
    if args.time_limit is not None:
        deadline = time.time() + args.time_limit * 3600
        print(f"Time limit: {args.time_limit}h")

    epochs_no_improve = 0

    # ── Main training loop ────────────────────────────────────
    print(f"\nFine-tuning for {args.epochs} epochs (start={start_epoch + 1})")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs):

        # Time limit check
        if deadline is not None and time.time() >= deadline:
            print(f"\nTime limit reached at epoch {epoch + 1}. Saving checkpoint.")
            _save_checkpoint(ckpt_dir, "latest", epoch - 1, model, optimizer,
                             scheduler, best_map, loss_history, map_history)
            break

        # Backbone freeze/unfreeze schedule
        if epoch == 0:
            _set_backbone_frozen(model, frozen=True)
            print(f"  [Backbone FROZEN for first {args.freeze_backbone_epochs} epochs]")
        elif epoch == args.freeze_backbone_epochs:
            _set_backbone_frozen(model, frozen=False)
            # Rebuild optimizer to include newly unfrozen params
            optimizer = torch.optim.AdamW(
                _get_trainable_params(), lr=args.lr * 0.1,
                weight_decay=args.weight_decay
            )
            print(f"  [Backbone UNFROZEN (last 2 blocks) at epoch {epoch + 1}]")

        model.train()
        t0 = time.time()
        epoch_losses = {"total": 0.0, "box": 0.0, "cls": 0.0,
                        "attn": 0.0, "contrast": 0.0, "physics": 0.0}
        n_batches = len(train_loader)

        for step, (images, targets) in enumerate(train_loader):
            # Linear LR warmup
            if epoch < args.warmup_epochs:
                total_ws = args.warmup_epochs * n_batches
                cur = epoch * n_batches + step
                scale = (cur + 1) / max(1, total_ws)
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * scale

            images = images.to(device)

            # Class-aware copy-paste (batch-level augmentation)
            images, targets = copy_paste.apply_batch(images, targets)

            # Forward
            outputs = model(images, training=True)

            # Target assignment
            pred_boxes, gt_boxes, class_ids, pred_cls, gt_cls = (
                target_assigner.assign(outputs, targets, device)
            )

            if pred_boxes is None or pred_boxes.numel() == 0:
                continue

            # Align fraud_implied with pred_cls dimensions
            # fraud_implied: (D, 6) from model detections
            # pred_cls: (N, 6) from target assigner (N = total GT boxes)
            # Fix: average fraud_implied to (1,6), expand to (N,6)
            # Same averaging method target assigner uses for pred_cls
            _fraud = outputs.get("fraud_implied")
            if _fraud is not None and pred_cls is not None:
                if _fraud.shape[0] != pred_cls.shape[0]:
                    _fraud = _fraud.mean(dim=0, keepdim=True).expand(pred_cls.shape[0], -1)

            # Loss
            losses = loss_fn(
                pred_boxes, gt_boxes, class_ids, pred_cls, gt_cls,
                attn_maps=outputs.get("attn_maps"),
                gt_boxes_list=[t["boxes"].to(device) for t in targets],
                gt_classes_list=[t["classes"].to(device) for t in targets],
                physics_implied=_fraud,
                predicted_class_logits=pred_cls,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k] += losses[k].item()

        if epoch >= args.warmup_epochs:
            scheduler.step()

        avg = {k: v / max(1, n_batches) for k, v in epoch_losses.items()}
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        # ── Validation ────────────────────────────────────────
        val_metrics = validate(model, val_loader, device)
        val_map = val_metrics["mAP"]
        map_history.append(val_map)

        # Record loss history
        row = {**avg, "epoch": epoch + 1, "val_map": val_map, "lr": lr_now}
        loss_history.append(row)

        # ── Print epoch summary ───────────────────────────────
        print(
            f"Epoch {epoch+1:4d}/{args.epochs}  "
            f"loss={avg['total']:.4f}  "
            f"box={avg['box']:.3f}  cls={avg['cls']:.3f}  "
            f"attn={avg['attn']:.4f}  "
            f"mAP@0.5={val_map:.4f}  "
            f"lr={lr_now:.2e}  {elapsed:.0f}s"
        )

        # Per-class AP on validation
        per_cls_str = "  ".join(
            f"{c}={val_metrics.get(c, 0):.3f}" for c in CLASS_NAMES
        )
        print(f"  Per-class AP: {per_cls_str}")
        if "mAP@0.5:0.95" in val_metrics:
            print(f"  mAP@0.5:0.95 = {val_metrics['mAP@0.5:0.95']:.4f}")

        # ── Milestone alerts ──────────────────────────────────
        for threshold, label in MILESTONES:
            if val_map > threshold and threshold not in milestones_beaten:
                milestones_beaten.add(threshold)
                print(f"\n  ★ MILESTONE: mAP {val_map:.4f} > {threshold} ({label}) ★\n")

        # ── Best model ────────────────────────────────────────
        is_best = val_map > best_map
        if is_best:
            best_map = val_map
            epochs_no_improve = 0
            _save_checkpoint(ckpt_dir, "best", epoch, model, optimizer,
                             scheduler, best_map, loss_history, map_history)
        else:
            epochs_no_improve += 1

        # Periodic checkpoint (CRITICAL for HPC job safety)
        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(ckpt_dir, "latest", epoch, model, optimizer,
                             scheduler, best_map, loss_history, map_history)
            _save_checkpoint(ckpt_dir, f"epoch_{epoch+1:04d}", epoch, model,
                             optimizer, scheduler, best_map, loss_history, map_history)

        # ── Early stopping ────────────────────────────────────
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch + 1} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # ── Final save & curves ───────────────────────────────────
    _save_checkpoint(ckpt_dir, "latest", epoch, model, optimizer,
                     scheduler, best_map, loss_history, map_history)

    # Save history as JSON
    (out_dir / "train_history.json").write_text(
        json.dumps({"loss_history": loss_history, "map_history": map_history,
                    "best_map": best_map}, indent=2)
    )

    _save_training_curves(loss_history, map_history, out_dir)

    print(f"\nTraining complete.")
    print(f"Best mAP@0.5: {best_map:.4f}")
    print(f"Best model:   {ckpt_dir / 'best.pt'}")


def _save_training_curves(
    loss_history: list[dict], map_history: list[float], out_dir: Path
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [r["epoch"] for r in loss_history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        # Loss
        ax1.plot(epochs, [r["total"] for r in loss_history], label="total", lw=1.5)
        ax1.plot(epochs, [r["box"]   for r in loss_history], label="box",   lw=1, ls="--")
        ax1.plot(epochs, [r["cls"]   for r in loss_history], label="cls",   lw=1, ls="--")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # mAP
        ax2.plot(range(1, len(map_history) + 1), map_history, color="green", lw=1.5)
        for thresh, lbl in MILESTONES:
            ax2.axhline(thresh, ls=":", color="gray", alpha=0.6)
            ax2.text(1, thresh + 0.003, lbl, fontsize=7, color="gray")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("mAP@0.5")
        ax2.set_title("Validation mAP@0.5")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        path = out_dir / "train_curves.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        print(f"  Training curves → {path}")
    except Exception as e:
        print(f"  (Could not save training curves: {e})")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune XCarDamageNet on CarDD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--data_dir",       required=True,
                   help="CarDD dataset root (must contain images/{train,val,test}/ "
                        "and labels/{train,val,test}/)")
    p.add_argument("--output_dir",     required=True,
                   help="Directory for logs, curves, and JSON history")
    p.add_argument("--checkpoint_dir", required=True,
                   help="Directory for .pt checkpoint files")

    # Optional
    p.add_argument("--resume",             default=None,
                   help="Checkpoint .pt to resume from")
    p.add_argument("--physics_checkpoint", default=None,
                   help="Pre-trained physics encoder .pt (from pretrain_physics.py)")
    p.add_argument("--config",             default=None,
                   help="YAML config file. Values override argument defaults.")

    # Training hyperparameters (spec defaults)
    p.add_argument("--epochs",                type=int,   default=150)
    p.add_argument("--patience",              type=int,   default=30)
    p.add_argument("--batch_size",            type=int,   default=8)
    p.add_argument("--lr",                    type=float, default=1e-4)
    p.add_argument("--weight_decay",          type=float, default=0.05)
    p.add_argument("--warmup_epochs",         type=int,   default=5)
    p.add_argument("--freeze_backbone_epochs",type=int,   default=30,
                   help="Keep DINOv2 backbone frozen for first N epochs")
    p.add_argument("--image_size",            type=int,   default=518)
    p.add_argument("--num_workers",           type=int,   default=4)
    p.add_argument("--save_every",            type=int,   default=10,
                   help="Save periodic checkpoint every N epochs (HPC safety)")
    p.add_argument("--time_limit",            type=float, default=None,
                   help="Stop after N hours. None = run until done.")
    p.add_argument("--amp",                   action="store_true", default=False,
                   help="Enable AMP (off by default — NaN risk on V100)")

    # Loss weights (CLAUDE.md spec: 7.5/0.5/1.5/0.10/0.05/0.02)
    p.add_argument("--loss_box",      type=float, default=7.5)
    p.add_argument("--loss_cls",      type=float, default=0.5)
    p.add_argument("--loss_dfl",      type=float, default=1.5)
    p.add_argument("--loss_attn",     type=float, default=0.10)
    p.add_argument("--loss_contrast", type=float, default=0.05)
    p.add_argument("--loss_physics",  type=float, default=0.02)

    # Augmentation (spec defaults)
    p.add_argument("--copy_paste",    type=float, default=0.30,
                   help="Class-aware copy-paste prob (rare classes only)")
    p.add_argument("--mosaic",        type=float, default=1.0)
    p.add_argument("--mixup",         type=float, default=0.10)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
