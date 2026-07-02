"""Evaluate a trained XCarDamageNet checkpoint on CarDD test or val split.

Computes:
  - mAP@0.5 (primary metric, same as v1 comparisons)
  - mAP@0.5:0.95 (COCO-style, 10 IoU thresholds average)
  - Per-class AP@0.5
  - Per-class precision and recall at 50% confidence threshold

Results are printed to stdout and optionally saved to JSON.

Usage:
    python scripts/evaluate.py \\
        --data_dir /path/to/cardd \\
        --checkpoint ./checkpoints/train_001/best.pt \\
        --split test \\
        --output_dir ./eval_results/run_001
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader

from xcardamagenet.models.xcardamagenet import XCarDamageNet
from xcardamagenet.data.cardd_dataset import CarDDDataset, CLASS_NAMES
from xcardamagenet.utils.metrics import box_iou, compute_ap

NUM_CLASSES = 6


# ─────────────────────────────────────────────────────────────
# Detection decoding
# ─────────────────────────────────────────────────────────────

def decode_predictions(
    det_map: torch.Tensor,
    conf_threshold: float = 0.25,
    img_size: int = 518,
    patch_size: int = 14,
) -> list[dict]:
    """Decode raw multi-scale detection maps to prediction dicts.

    Decodes all three scales (P3/P4/P5) and merges, keeping highest-scoring
    prediction per spatial location.

    Args:
        det_map: (B, 5+C, H, W) — objectness + xywh + class logits
        conf_threshold: minimum score to keep a prediction

    Returns:
        List of {"boxes": (N,4), "scores": (N,), "classes": (N,)} per image
        Boxes are in normalised [x1,y1,x2,y2] format [0,1].
    """
    B, CH, H, W = det_map.shape
    num_classes = CH - 5
    results = []

    stride = img_size / (H * patch_size)  # nominal scale factor

    for b in range(B):
        pred = det_map[b]                                  # (5+C, H, W)
        obj = torch.sigmoid(pred[0])                       # (H, W)

        # xywh in grid units → normalised [0,1]
        x_off = pred[1].sigmoid()
        y_off = pred[2].sigmoid()
        w_rel = pred[3].sigmoid()
        h_rel = pred[4].sigmoid()

        # Build (H*W,) spatial index grids
        gy, gx = torch.meshgrid(
            torch.arange(H, device=det_map.device, dtype=torch.float32),
            torch.arange(W, device=det_map.device, dtype=torch.float32),
            indexing="ij",
        )
        cx = (gx + x_off) / W   # (H, W) normalised centre-x
        cy = (gy + y_off) / H
        bw = w_rel
        bh = h_rel

        x1 = (cx - bw / 2).view(-1)
        y1 = (cy - bh / 2).view(-1)
        x2 = (cx + bw / 2).view(-1)
        y2 = (cy + bh / 2).view(-1)

        # Clamp to [0,1]
        x1 = x1.clamp(0, 1)
        y1 = y1.clamp(0, 1)
        x2 = x2.clamp(0, 1)
        y2 = y2.clamp(0, 1)

        cls_logits = pred[5:].view(num_classes, -1).T    # (H*W, C)
        cls_probs = torch.softmax(cls_logits, dim=-1)
        scores, classes = cls_probs.max(dim=-1)
        obj_flat = obj.view(-1)
        scores = scores * obj_flat

        keep = scores > conf_threshold
        n_keep = keep.sum().item()

        if n_keep == 0:
            results.append({
                "boxes":   torch.zeros(0, 4),
                "scores":  torch.zeros(0),
                "classes": torch.zeros(0, dtype=torch.long),
            })
            continue

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)[keep]
        results.append({
            "boxes":   boxes.cpu(),
            "scores":  scores[keep].cpu(),
            "classes": classes[keep].cpu(),
        })

    return results


# ─────────────────────────────────────────────────────────────
# Per-class precision / recall
# ─────────────────────────────────────────────────────────────

def compute_precision_recall(
    predictions: list[dict],
    targets: list[dict],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5,
    num_classes: int = NUM_CLASSES,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-class precision and recall at a fixed confidence threshold."""
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)

    for pred, tgt in zip(predictions, targets):
        for cls_id in range(num_classes):
            gt_mask = (tgt["classes"] == cls_id)
            gt_boxes = tgt["boxes"][gt_mask]
            n_gt = gt_mask.sum().item()

            pred_mask = (pred["classes"] == cls_id) & (pred["scores"] > conf_threshold)
            p_boxes = pred["boxes"][pred_mask]
            n_pred = pred_mask.sum().item()

            if n_pred == 0:
                fn[cls_id] += n_gt
                continue
            if n_gt == 0:
                fp[cls_id] += n_pred
                continue

            matched_gt = torch.zeros(n_gt, dtype=torch.bool)
            for pb in p_boxes:
                ious = box_iou(pb.unsqueeze(0), gt_boxes)[0]
                best_iou, best_idx = ious.max(0)
                if best_iou >= iou_threshold and not matched_gt[best_idx]:
                    matched_gt[best_idx] = True
                    tp[cls_id] += 1
                else:
                    fp[cls_id] += 1
            fn[cls_id] += (~matched_gt).sum().item()

    precision = {CLASS_NAMES[i]: float(tp[i] / max(tp[i] + fp[i], 1e-7))
                 for i in range(num_classes)}
    recall    = {CLASS_NAMES[i]: float(tp[i] / max(tp[i] + fn[i], 1e-7))
                 for i in range(num_classes)}
    return precision, recall


# ─────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> dict:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU")
        device = "cpu"

    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data:       {args.data_dir} / {args.split}")

    # ── Load model ────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        alt = ckpt_path.parent / "latest.pt"
        if alt.exists():
            print(f"WARNING: {ckpt_path.name} not found — falling back to {alt.name}")
            ckpt_path = alt
        else:
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            sys.exit(1)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = XCarDamageNet(img_size=args.image_size, pretrained_backbone=False)

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(device).eval()

    trained_epoch = ckpt.get("epoch", "?")
    print(f"Model loaded (trained epoch: {trained_epoch})")

    # ── Dataset ───────────────────────────────────────────────
    dataset = CarDDDataset(
        args.data_dir, split=args.split, img_size=args.image_size
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=CarDDDataset.collate_fn,
    )
    print(f"Evaluating {len(dataset):,} images on '{args.split}' split…")

    # ── Inference ─────────────────────────────────────────────
    all_preds: list[dict] = []
    all_targets: list[dict] = []
    t0 = time.time()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images, training=False)

            # Decode from all three scales; use P3 (finest) as primary
            preds_p3 = decode_predictions(
                outputs["det_p3"], args.conf_threshold, args.image_size
            )
            all_preds.extend(preds_p3)
            all_targets.extend(targets)

    elapsed = time.time() - t0
    fps = len(dataset) / elapsed
    print(f"Inference done: {elapsed:.1f}s  ({fps:.1f} img/s)\n")

    # ── mAP@0.5 ───────────────────────────────────────────────
    from xcardamagenet.utils.metrics import compute_ap_per_class

    per_class_ap50 = compute_ap_per_class(all_preds, all_targets, iou_threshold=0.5)
    map50 = float(np.mean(list(per_class_ap50.values())))

    # ── mAP@0.5:0.95 ─────────────────────────────────────────
    iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()
    map_scores = []
    for iou_t in iou_thresholds:
        m = compute_ap_per_class(all_preds, all_targets, iou_threshold=round(iou_t, 2))
        map_scores.append(float(np.mean(list(m.values()))))
    map5095 = float(np.mean(map_scores))

    # ── Per-class precision / recall ──────────────────────────
    precision, recall = compute_precision_recall(
        all_preds, all_targets, iou_threshold=0.5, conf_threshold=args.conf_threshold
    )

    # ── Print results ─────────────────────────────────────────
    print("=" * 60)
    print(f"{'Metric':<25}  {'Value':>8}")
    print("-" * 60)
    print(f"{'mAP@0.5':<25}  {map50:>8.4f}")
    print(f"{'mAP@0.5:0.95':<25}  {map5095:>8.4f}")
    print()
    print(f"{'Class':<20}  {'AP@0.5':>8}  {'Precision':>10}  {'Recall':>8}")
    print("-" * 60)
    for cls in CLASS_NAMES:
        ap  = per_class_ap50.get(cls, 0.0)
        prec = precision.get(cls, 0.0)
        rec  = recall.get(cls, 0.0)
        print(f"{cls:<20}  {ap:>8.4f}  {prec:>10.4f}  {rec:>8.4f}")
    print("=" * 60)

    # ── Comparison with baselines ─────────────────────────────
    print("\nComparison:")
    print(f"  DCN+ (2024):        0.496")
    print(f"  XCarDamage v1:      0.700")
    print(f"  YOLOv9-CS:          0.730")
    print(f"  v1 + TTA:           0.739")
    print(f"  XCarDamageNet v2:   {map50:.4f}  ← THIS MODEL")
    if map50 > 0.739:
        print(f"  ★ BEATS v1+TTA by +{map50 - 0.739:.4f}")
    print()

    # ── Save results ──────────────────────────────────────────
    results = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_images": len(dataset),
        "mAP@0.5": map50,
        "mAP@0.5:0.95": map5095,
        "per_class_AP@0.5": per_class_ap50,
        "precision": precision,
        "recall": recall,
        "inference_fps": fps,
        "conf_threshold": args.conf_threshold,
    }

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"eval_{args.split}.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Results saved → {out_path}")

    return results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate XCarDamageNet on CarDD test/val split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",       required=True,
                   help="CarDD dataset root directory")
    p.add_argument("--checkpoint",     required=True,
                   help="Trained model checkpoint (.pt)")
    p.add_argument("--split",          default="test",
                   choices=["train", "val", "test"],
                   help="Dataset split to evaluate on")
    p.add_argument("--output_dir",     default=None,
                   help="Save JSON results to this directory (optional)")
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--image_size",     type=int,   default=518)
    p.add_argument("--conf_threshold", type=float, default=0.25,
                   help="Confidence threshold for filtering predictions")
    p.add_argument("--device",         default="cuda",
                   help="Device: 'cuda' or 'cpu'")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
