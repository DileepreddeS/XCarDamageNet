"""Validator for XCarDamageNet — computes mAP and per-class AP on CarDD."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from typing import Dict, List


class Validator:
    """Evaluates XCarDamageNet on a validation/test split.

    Computes:
    - mAP@0.5 overall
    - AP@0.5 per class
    - Mean severity error
    """

    CLASS_NAMES = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]

    def __init__(
        self,
        model,
        val_loader: DataLoader,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device)
        self.val_loader = val_loader
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run full validation pass.

        Returns:
            dict with mAP and per-class APs.
        """
        self.model.eval()
        all_preds: List[dict] = []
        all_targets: List[dict] = []

        for images, targets in self.val_loader:
            images = images.to(self.device)
            outputs = self.model(images, training=False)
            all_targets.extend(targets)

            # Decode P3 predictions (finest scale, best for small damage)
            preds = self._decode_predictions(outputs["det_p3"])
            all_preds.extend(preds)

        # Simplified mAP computation (full implementation in utils/metrics.py)
        metrics = self._compute_map(all_preds, all_targets)
        return metrics

    def _decode_predictions(
        self, det_map: torch.Tensor
    ) -> List[dict]:
        """Convert raw detection map to list of prediction dicts.

        Args:
            det_map: (B, 5+C, H, W) — obj + box + class logits
        Returns:
            List of {"boxes", "scores", "classes"} per image.
        """
        B = det_map.shape[0]
        results = []

        for b in range(B):
            pred = det_map[b]  # (5+C, H, W)
            obj = torch.sigmoid(pred[0])       # (H, W)
            boxes = pred[1:5]                  # (4, H, W) — xywh
            cls_logits = pred[5:]              # (C, H, W)
            cls_probs = torch.softmax(cls_logits, dim=0)

            # Flatten spatial dims
            obj_flat = obj.view(-1)
            cls_flat = cls_probs.view(cls_probs.shape[0], -1).T  # (N, C)
            scores, classes = cls_flat.max(dim=-1)
            scores = scores * obj_flat

            # Filter by confidence
            keep = scores > self.conf_threshold
            results.append({
                "boxes": boxes.view(4, -1).T[keep],   # (n, 4)
                "scores": scores[keep],
                "classes": classes[keep],
            })

        return results

    def _compute_map(
        self, predictions: List[dict], targets: List[dict]
    ) -> Dict[str, float]:
        """Compute mAP@0.5 (simplified — full implementation in utils/metrics.py)."""
        # Placeholder: returns zeros until utils/metrics.py is integrated
        return {
            "mAP@0.5": 0.0,
            **{f"AP_{c}@0.5": 0.0 for c in self.CLASS_NAMES},
        }
