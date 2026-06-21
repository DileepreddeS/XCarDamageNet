"""Export XCarDamageNet to ONNX or TensorRT for production deployment."""

from __future__ import annotations

import torch
from pathlib import Path


def export_onnx(
    model,
    output_path: str,
    img_size: int = 518,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> None:
    """Export model to ONNX format.

    Args:
        model: Trained XCarDamageNet model.
        output_path: Path to save .onnx file.
        img_size: Input image size.
        opset_version: ONNX opset.
        dynamic_batch: Allow dynamic batch size.
    """
    model.eval()
    dummy = torch.zeros(1, 3, img_size, img_size)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch_size"},
        }

    output_path = str(Path(output_path).with_suffix(".onnx"))
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["images"],
        output_names=["det_p3", "det_p4", "det_p5"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"ONNX model exported to {output_path}")
