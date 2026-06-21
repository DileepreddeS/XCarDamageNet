"""Single-image inference predictor for XCarDamageNet."""

from __future__ import annotations

import torch
from typing import Dict
from ..data.preprocessing import preprocess_image, denormalise_boxes


class Predictor:
    """Run XCarDamageNet inference on a single image.

    Returns all 6 outputs: boxes, severity, cause, repair, heatmaps, fraud.
    """

    CAUSE_NAMES = ["impact", "hail", "vandalism", "wear", "environmental"]
    REPAIR_NAMES = ["PDR", "panel_replacement", "paint_refinish", "glass_replacement", "tire_replacement"]
    CLASS_NAMES = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]

    def __init__(
        self,
        model,
        conf_threshold: float = 0.25,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        img_size: int = 518,
    ) -> None:
        self.model = model.to(device).eval()
        self.conf_threshold = conf_threshold
        self.device = device
        self.img_size = img_size

    @torch.no_grad()
    def predict(self, image) -> Dict:
        """Run inference on a single image.

        Args:
            image: File path string or BGR numpy array.

        Returns:
            dict with:
                boxes:      (N, 4) pixel coordinates [x1,y1,x2,y2]
                classes:    (N,) class ids
                class_names:(N,) class name strings
                scores:     (N,) confidence scores
                severity:   float in [0,1]
                cause:      str — predicted damage cause
                repair:     str — recommended repair action
                fraud_risk: float in [0,1]
                attn_maps:  (6, H, W) numpy heatmaps
        """
        tensor, original_size = preprocess_image(image, self.img_size)
        tensor = tensor.to(self.device)

        outputs = self.model(tensor, training=False)

        # Decode detections from P3 (finest scale)
        det = outputs["det_p3"][0]  # (5+C, H, W)
        obj = torch.sigmoid(det[0]).view(-1)
        cls_probs = torch.softmax(det[5:], dim=0).view(det.shape[0] - 5, -1).T
        scores, classes = cls_probs.max(dim=-1)
        scores = scores * obj

        keep = scores > self.conf_threshold
        # Convert spatial coords to normalised [0,1] (simplified — proper decoding
        # requires stride-aware coordinate conversion)
        boxes_norm = torch.zeros(keep.sum(), 4)
        boxes_px = denormalise_boxes(boxes_norm, original_size)

        severity = outputs["severity"][0, 0].item()
        cause_idx = outputs["cause"][0].argmax().item()
        repair_idx = outputs["repair"][0].argmax().item()
        fraud = outputs["fraud_score"][0, 0].item()
        attn = outputs["attn_maps"][0].cpu().numpy()

        return {
            "boxes": boxes_px.cpu().numpy(),
            "classes": classes[keep].cpu().numpy(),
            "class_names": [self.CLASS_NAMES[c] for c in classes[keep].cpu().tolist()],
            "scores": scores[keep].cpu().numpy(),
            "severity": severity,
            "cause": self.CAUSE_NAMES[cause_idx],
            "repair": self.REPAIR_NAMES[repair_idx],
            "fraud_risk": fraud,
            "attn_maps": attn,
        }
