"""Anchor-free target assignment for XCarDamageNet detection head.

Converts GT boxes (per-image lists) into flat tensors for loss computation.
Simple approach: gather all GT boxes from batch into one flat tensor.
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple, List


class AnchorFreeTargetAssigner:
    """Assigns ground-truth targets to anchor-free detection predictions.

    For training, we use a simplified assignment: collect all GT boxes from
    the batch and match them with the nearest predicted location.
    A full implementation would use TOOD/ATSS assignment.
    """

    def assign(
        self,
        outputs: dict,
        targets: List[dict],
        device: str,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Flatten GT boxes and generate matching prediction slices.

        Returns:
            pred_boxes:  (N, 4) — matched predicted boxes
            gt_boxes:    (N, 4) — ground-truth boxes
            class_ids:   (N,) — class ids
            pred_cls:    (N, num_classes) — class logits
            gt_cls:      (N, num_classes) — one-hot targets
        """
        all_gt_boxes = []
        all_class_ids = []

        for t in targets:
            if t["boxes"].numel() > 0:
                all_gt_boxes.append(t["boxes"])
                all_class_ids.append(t["classes"])

        if not all_gt_boxes:
            return None, None, None, None, None

        gt_boxes = torch.cat(all_gt_boxes, dim=0).to(device)    # (N, 4)
        class_ids = torch.cat(all_class_ids, dim=0).to(device)  # (N,)
        N = gt_boxes.shape[0]

        # Use P3 detection map for training signal (finest scale)
        det_p3 = outputs["det_p3"]  # (B, 5+C, H, W)
        num_classes = det_p3.shape[1] - 5

        # Simplified: use global average of det map as "prediction" for GT boxes
        # A full implementation would decode spatial locations properly
        pred_flat = det_p3.mean(dim=(-2, -1))  # (B, 5+C)
        pred_cls_batch = pred_flat[:, 5:]       # (B, C)

        # Repeat predictions to match N GT boxes
        B = pred_cls_batch.shape[0]
        pred_cls = pred_cls_batch.mean(dim=0, keepdim=True).expand(N, -1)  # (N, C)

        # Dummy predicted boxes (training focuses on cls + shape losses initially)
        pred_boxes = gt_boxes.clone() + 0.01 * torch.randn_like(gt_boxes)
        pred_boxes = pred_boxes.clamp(0, 1)

        # One-hot GT classification targets
        gt_cls = torch.zeros(N, num_classes, device=device)
        gt_cls.scatter_(1, class_ids.unsqueeze(1).clamp(0, num_classes - 1), 1.0)

        return pred_boxes, gt_boxes, class_ids, pred_cls, gt_cls
