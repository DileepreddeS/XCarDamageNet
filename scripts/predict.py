"""Single-image inference with XCarDamageNet.

Loads a trained model checkpoint and runs inference on one image.
Saves:
  - {stem}_annotated.jpg  — original image with bounding boxes, class labels,
                            severity score, cause, repair action, fraud risk
  - {stem}_heatmaps.jpg   — grid of 6 per-class attention heatmaps overlaid

All 6 model outputs are printed to stdout.

Usage:
    python scripts/predict.py \\
        --image ./car.jpg \\
        --checkpoint ./checkpoints/train_001/best.pt \\
        --output_dir ./predictions \\
        --conf_threshold 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from xcardamagenet.models.xcardamagenet import XCarDamageNet
from xcardamagenet.data.preprocessing import preprocess_image
from xcardamagenet.utils.visualization import draw_heatmap

CLASS_NAMES   = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]
CAUSE_NAMES   = ["impact", "hail", "vandalism", "wear", "environmental"]
REPAIR_NAMES  = ["PDR", "panel_replacement", "paint_refinish",
                 "glass_replacement", "tire_replacement"]

# Per-class colours (BGR)
COLORS = [
    (100, 100, 255),  # dent          — red
    (100, 200, 100),  # scratch        — green
    ( 50,  50, 200),  # crack          — dark red
    (200, 200,  50),  # glass_shatter  — cyan-ish
    (200,  50, 200),  # lamp_broken    — magenta
    ( 50, 200, 200),  # tire_flat      — yellow
]


# ─────────────────────────────────────────────────────────────
# Decoding
# ─────────────────────────────────────────────────────────────

def decode_detections(
    det_map: torch.Tensor,
    conf_threshold: float,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a single detection map to pixel-space boxes.

    Args:
        det_map: (5+C, H, W) — for a single image
        conf_threshold: minimum score
        img_size: original image size (assumed square) for coordinate scaling

    Returns:
        boxes:   (N, 4) float32 pixel coords [x1, y1, x2, y2]
        scores:  (N,) float32
        classes: (N,) int32
    """
    CH, H, W = det_map.shape
    num_classes = CH - 5

    obj   = det_map[0].sigmoid()                           # (H, W)
    x_off = det_map[1].sigmoid()
    y_off = det_map[2].sigmoid()
    bw    = det_map[3].sigmoid()
    bh    = det_map[4].sigmoid()

    gy, gx = torch.meshgrid(
        torch.arange(H, device=det_map.device, dtype=torch.float32),
        torch.arange(W, device=det_map.device, dtype=torch.float32),
        indexing="ij",
    )
    cx = (gx + x_off) / W     # normalised centre-x
    cy = (gy + y_off) / H

    x1 = ((cx - bw / 2) * img_size).clamp(0, img_size)
    y1 = ((cy - bh / 2) * img_size).clamp(0, img_size)
    x2 = ((cx + bw / 2) * img_size).clamp(0, img_size)
    y2 = ((cy + bh / 2) * img_size).clamp(0, img_size)

    cls_logits = det_map[5:].view(num_classes, -1).T   # (H*W, C)
    cls_probs  = torch.softmax(cls_logits, dim=-1)
    scores, classes = cls_probs.max(dim=-1)
    scores = scores * obj.view(-1)

    keep = scores > conf_threshold

    if keep.sum() == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )

    boxes_np  = torch.stack([x1.view(-1), y1.view(-1),
                              x2.view(-1), y2.view(-1)], dim=-1)[keep].cpu().numpy()
    scores_np = scores[keep].cpu().numpy()
    cls_np    = classes[keep].cpu().numpy().astype(np.int32)

    # Simple NMS (greedy by score)
    order = scores_np.argsort()[::-1]
    kept  = []
    suppressed = np.zeros(len(order), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        kept.append(i)
        bi = boxes_np[i]
        for j in order:
            if suppressed[j] or j == i:
                continue
            bj = boxes_np[j]
            inter_x1 = max(bi[0], bj[0]); inter_y1 = max(bi[1], bj[1])
            inter_x2 = min(bi[2], bj[2]); inter_y2 = min(bi[3], bj[3])
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue
            inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
            area_j = (bj[2] - bj[0]) * (bj[3] - bj[1])
            iou = inter / max(area_i + area_j - inter, 1e-7)
            if iou > 0.45:
                suppressed[j] = True

    kept = np.array(kept)
    return boxes_np[kept], scores_np[kept], cls_np[kept]


# ─────────────────────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────────────────────

def annotate_image(
    img: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    severity: float,
    cause: str,
    repair: str,
    fraud_risk: float,
) -> np.ndarray:
    """Draw all predictions on a copy of the image (BGR)."""
    out = img.copy()

    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
        x1, y1, x2, y2 = box.astype(int)
        cls_id = int(cls_id)
        color = COLORS[cls_id % len(COLORS)]

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 5)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Summary panel (top-left)
    lines = [
        f"Severity: {severity:.2f}",
        f"Cause:    {cause}",
        f"Repair:   {repair}",
    ]
    if fraud_risk > 0.4:
        lines.append(f"FRAUD RISK: {fraud_risk:.2f}")

    y_off = 22
    for line in lines:
        color = (0, 0, 220) if "FRAUD" in line else (50, 50, 50)
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (8, y_off - th - 3), (8 + tw + 4, y_off + 3), (230, 230, 230), -1)
        cv2.putText(out, line, (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y_off += th + 8

    return out


def make_heatmap_grid(
    img: np.ndarray,
    attn_maps: np.ndarray,
    img_size: int,
) -> np.ndarray:
    """Produce a 2×3 grid of class-specific attention map overlays."""
    panels = []
    for i, cls_name in enumerate(CLASS_NAMES):
        hmap = attn_maps[i]                              # (H, W) float32 [0,1]
        panel = draw_heatmap(img, hmap, alpha=0.5)
        h, w = panel.shape[:2]
        # Label
        cv2.putText(panel, cls_name, (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)

    # Arrange in 2 rows × 3 columns
    row0 = np.concatenate(panels[:3], axis=1)
    row1 = np.concatenate(panels[3:], axis=1)
    return np.concatenate([row0, row1], axis=0)


# ─────────────────────────────────────────────────────────────
# Main inference
# ─────────────────────────────────────────────────────────────

def predict(args: argparse.Namespace) -> None:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU")
        device = "cpu"

    # ── Load model ────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = XCarDamageNet(img_size=args.image_size, pretrained_backbone=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"  Model ready on {device}")

    # ── Load image ────────────────────────────────────────────
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"ERROR: Image not found: {img_path}")
        sys.exit(1)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"ERROR: Could not read image: {img_path}")
        sys.exit(1)

    original_h, original_w = img_bgr.shape[:2]
    print(f"Image: {img_path.name}  ({original_w}×{original_h})")

    # Preprocess
    tensor, orig_size = preprocess_image(img_bgr, args.image_size)
    tensor = tensor.to(device)

    # ── Inference ─────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(tensor, training=False)

    # ── Decode detections ─────────────────────────────────────
    det_map = outputs["det_p3"][0]  # (5+C, H, W)
    boxes, scores, classes = decode_detections(
        det_map, args.conf_threshold, args.image_size
    )

    # Scale boxes from image_size coords back to original size
    if boxes.shape[0] > 0:
        scale_x = original_w / args.image_size
        scale_y = original_h / args.image_size
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y

    # ── Extract scalar outputs ────────────────────────────────
    severity  = float(outputs["severity"][0, 0].item())
    cause_idx = int(outputs["cause"][0].argmax().item())
    repair_idx= int(outputs["repair"][0].argmax().item())
    fraud     = float(outputs["fraud_score"][0, 0].item())
    attn_maps = outputs["attn_maps"][0].cpu().numpy()   # (6, H, W)

    cause_name  = CAUSE_NAMES[cause_idx]
    repair_name = REPAIR_NAMES[repair_idx]

    # Resize attn_maps to original image size for overlay
    attn_resized = []
    for hmap in attn_maps:
        h_t = torch.from_numpy(hmap).unsqueeze(0).unsqueeze(0)
        h_up = F.interpolate(h_t, size=(original_h, original_w),
                             mode="bilinear", align_corners=False)
        attn_resized.append(h_up.squeeze().numpy())
    attn_resized = np.stack(attn_resized)   # (6, H_orig, W_orig)

    # ── Print results ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Detections: {len(boxes)}")
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
        print(f"    [{i+1}] {CLASS_NAMES[int(cls_id)]:<15} "
              f"score={score:.3f}  "
              f"box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

    print(f"\n  Severity:   {severity:.3f}  (0=minor, 1=severe)")
    print(f"  Cause:      {cause_name}")
    print(f"  Repair:     {repair_name}")
    print(f"  Fraud risk: {fraud:.3f}  {'⚠ HIGH' if fraud > 0.5 else '(normal)'}")

    anomaly_mean = float(outputs["anomaly_scores"][0].mean().item())
    damage_mean  = float(outputs["damage_scores"][0].mean().item())
    print(f"\n  Anomaly score (mean): {anomaly_mean:.3f}")
    print(f"  Damage  score (mean): {damage_mean:.3f}")
    print("=" * 55)

    # ── Save outputs ──────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    # Annotated image
    annotated = annotate_image(
        img_bgr, boxes, scores, classes,
        severity, cause_name, repair_name, fraud
    )
    ann_path = out_dir / f"{stem}_annotated.jpg"
    cv2.imwrite(str(ann_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\nAnnotated image → {ann_path}")

    # Heatmap grid
    # Resize original to inference size for heatmap overlay
    img_resized = cv2.resize(img_bgr, (original_w, original_h))
    heatmap_grid = make_heatmap_grid(img_resized, attn_resized, original_w)
    hmap_path = out_dir / f"{stem}_heatmaps.jpg"
    cv2.imwrite(str(hmap_path), heatmap_grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Heatmap grid   → {hmap_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="XCarDamageNet single-image inference with visualisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image",          required=True,
                   help="Path to input image (JPG/PNG)")
    p.add_argument("--checkpoint",     required=True,
                   help="Trained model checkpoint (.pt)")
    p.add_argument("--output_dir",     default="./predictions",
                   help="Directory to save annotated images and heatmaps")
    p.add_argument("--device",         default="cuda",
                   help="Device: 'cuda' or 'cpu'")
    p.add_argument("--conf_threshold", type=float, default=0.25,
                   help="Detection confidence threshold")
    p.add_argument("--image_size",     type=int,   default=518,
                   help="Inference image size (must match training)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args)
