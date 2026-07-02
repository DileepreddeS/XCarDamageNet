"""Anchor-free target assignment for XCarDamageNet detection head.

Uses center-based assignment on the P3 (finest scale) detection map:
  - For each GT box, find all grid cells whose centers fall inside the GT box.
  - Those cells are positive samples; their decoded predictions are matched
    to the GT box.
  - If no cell center falls inside (very small GT box), fall back to the
    single closest cell center.

Detection map format (B, 5+C, H, W):
  ch[0]      — objectness logit (raw)
  ch[1..4]   — box offsets (raw): sigmoid → cx/cy offset within cell, bw/bh
  ch[5..]    — class logits (raw)

Box decoding (P3 grid H×W):
  cx = (col + sigmoid(ch[1])) / W   (normalised [0,1])
  cy = (row + sigmoid(ch[2])) / H
  bw = sigmoid(ch[3])               (normalised [0,1])
  bh = sigmoid(ch[4])
  x1 = cx - bw/2,  x2 = cx + bw/2
  y1 = cy - bh/2,  y2 = cy + bh/2
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple, List


class AnchorFreeTargetAssigner:
    """Assigns ground-truth targets to anchor-free P3 detection predictions."""

    def assign(
        self,
        outputs: dict,
        targets: List[dict],
        device: str,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Center-based spatial target assignment.

        Args:
            outputs: model output dict (must contain "det_p3")
            targets: list of per-image dicts with "boxes" (M,4) [x1,y1,x2,y2]
                     normalised [0,1] and "classes" (M,) int
            device:  target device string

        Returns:
            pred_boxes:  (N, 4) — decoded predicted boxes [x1,y1,x2,y2]
            gt_boxes:    (N, 4) — matched GT boxes
            class_ids:   (N,) — integer class labels
            pred_cls:    (N, C) — class logits from assigned cells
            gt_cls:      (N, C) — one-hot class targets
            (all None when batch has no GT boxes)
        """
        det_p3 = outputs["det_p3"]                  # (B, 5+C, H, W)
        B, CH, H, W = det_p3.shape
        num_classes = CH - 5

        # Precompute grid cell centres once, on the same device as det_p3
        dev = det_p3.device
        rows = torch.arange(H, device=dev, dtype=torch.float32)
        cols = torch.arange(W, device=dev, dtype=torch.float32)
        gy, gx = torch.meshgrid(rows, cols, indexing="ij")
        cx_grid = (gx + 0.5) / W   # (H, W) normalised centre-x
        cy_grid = (gy + 0.5) / H   # (H, W) normalised centre-y
        cx_flat = cx_grid.view(-1)  # (H*W,)
        cy_flat = cy_grid.view(-1)
        gx_flat = gx.view(-1)       # grid column index
        gy_flat = gy.view(-1)       # grid row index

        all_pred_boxes: list[torch.Tensor] = []
        all_gt_boxes:   list[torch.Tensor] = []
        all_class_ids:  list[torch.Tensor] = []
        all_pred_cls:   list[torch.Tensor] = []

        for b in range(B):
            tgt = targets[b]
            if tgt["boxes"].numel() == 0:
                continue

            gt_b  = tgt["boxes"].to(dev)    # (M, 4) [x1,y1,x2,y2] normalised
            cls_b = tgt["classes"].to(dev)  # (M,)

            pred_b = det_p3[b]              # (CH, H, W)

            # ── Decode predicted boxes for every grid cell ────────────
            x_off = torch.sigmoid(pred_b[1]).view(-1)   # (H*W,) ∈ [0,1]
            y_off = torch.sigmoid(pred_b[2]).view(-1)
            bw    = torch.sigmoid(pred_b[3]).view(-1)
            bh    = torch.sigmoid(pred_b[4]).view(-1)

            cx_pred = (gx_flat + x_off) / W   # normalised [0,1]
            cy_pred = (gy_flat + y_off) / H

            pred_x1 = (cx_pred - bw / 2).clamp(0.0, 1.0)
            pred_y1 = (cy_pred - bh / 2).clamp(0.0, 1.0)
            pred_x2 = (cx_pred + bw / 2).clamp(0.0, 1.0)
            pred_y2 = (cy_pred + bh / 2).clamp(0.0, 1.0)
            pred_boxes_all = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)  # (H*W, 4)

            # ── Class logits for every cell ───────────────────────────
            pred_cls_all = pred_b[5:].view(num_classes, -1).T    # (H*W, C)

            # ── Per-GT-box cell assignment ────────────────────────────
            for m in range(gt_b.shape[0]):
                box = gt_b[m]  # [x1, y1, x2, y2]

                # Cells whose centers fall inside the GT box
                inside = (
                    (cx_flat >= box[0]) & (cx_flat <= box[2]) &
                    (cy_flat >= box[1]) & (cy_flat <= box[3])
                )  # (H*W,) bool

                if inside.sum() == 0:
                    # Fallback: single closest cell to GT box centre
                    gt_cx = (box[0] + box[2]) * 0.5
                    gt_cy = (box[1] + box[3]) * 0.5
                    dist = (cx_flat - gt_cx).pow(2) + (cy_flat - gt_cy).pow(2)
                    best = dist.argmin()
                    inside = torch.zeros(H * W, dtype=torch.bool, device=dev)
                    inside[best] = True

                n_pos = inside.sum().item()
                all_pred_boxes.append(pred_boxes_all[inside])                   # (n_pos, 4)
                all_gt_boxes.append(box.unsqueeze(0).expand(n_pos, -1))         # (n_pos, 4)
                all_class_ids.append(cls_b[m].unsqueeze(0).expand(n_pos))       # (n_pos,)
                all_pred_cls.append(pred_cls_all[inside])                        # (n_pos, C)

        if not all_pred_boxes:
            return None, None, None, None, None

        pred_boxes = torch.cat(all_pred_boxes, dim=0).to(device)    # (N, 4)
        gt_boxes   = torch.cat(all_gt_boxes,   dim=0).to(device)    # (N, 4)
        class_ids  = torch.cat(all_class_ids,  dim=0).to(device)    # (N,)
        pred_cls   = torch.cat(all_pred_cls,   dim=0).to(device)    # (N, C)

        N = class_ids.shape[0]
        gt_cls = torch.zeros(N, num_classes, device=device)
        gt_cls.scatter_(1, class_ids.unsqueeze(1).clamp(0, num_classes - 1), 1.0)

        return pred_boxes, gt_boxes, class_ids, pred_cls, gt_cls
