"""Tests for DINOv2Backbone module.

Verifies: correct output shapes, frozen/unfrozen parameter counts,
and that forward pass runs without errors.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from xcardamagenet.models.backbone import DINOv2Backbone


def test_output_shape_default():
    """Input (B, 3, 518, 518) → output (B, N, 384)."""
    model = DINOv2Backbone(pretrained=False)
    x = torch.randn(2, 3, 518, 518)
    tokens = model(x)

    H_p, W_p = 518 // 14, 518 // 14  # 37x37 = 1369
    assert tokens.shape == (2, H_p * W_p, 384), (
        f"Expected (2, {H_p*W_p}, 384), got {tokens.shape}"
    )


def test_output_shape_custom_size():
    """Test with non-default image size (448x448)."""
    model = DINOv2Backbone(pretrained=False, img_size=448)
    x = torch.randn(1, 3, 448, 448)
    tokens = model(x)

    H_p, W_p = 448 // 14, 448 // 14  # 32x32 = 1024
    assert tokens.shape == (1, H_p * W_p, 384), (
        f"Expected (1, {H_p*W_p}, 384), got {tokens.shape}"
    )


def test_frozen_by_default():
    """With unfreeze_last_n_blocks=0, no parameters should be trainable."""
    model = DINOv2Backbone(pretrained=False, unfreeze_last_n_blocks=0)
    assert model.trainable_parameters() == 0, (
        f"Expected 0 trainable params, got {model.trainable_parameters()}"
    )


def test_unfreeze_last_2_blocks():
    """Unfreezing last 2 blocks should expose some trainable params."""
    model = DINOv2Backbone(pretrained=False, unfreeze_last_n_blocks=2)
    trainable = model.trainable_parameters()
    assert trainable > 0, "Expected some trainable params after unfreezing 2 blocks"
    assert trainable < model.total_parameters(), (
        "Expected fewer trainable params than total (backbone should be mostly frozen)"
    )
    print(f"  Trainable: {trainable:,} / {model.total_parameters():,}")


def test_grid_size():
    """Grid size should match (H/14, W/14)."""
    model = DINOv2Backbone(pretrained=False, img_size=518)
    assert model.get_grid_size() == (37, 37)


def test_embed_dim():
    """Embed dim should be 384 for ViT-S."""
    model = DINOv2Backbone(pretrained=False)
    assert model.embed_dim == 384


def test_no_cls_token_in_output():
    """Output token count should equal patch count, not patch+1."""
    model = DINOv2Backbone(pretrained=False, img_size=518)
    x = torch.randn(1, 3, 518, 518)
    tokens = model(x)
    expected_n = 37 * 37  # 1369 patches, no CLS
    assert tokens.shape[1] == expected_n, (
        f"Expected {expected_n} tokens (no CLS), got {tokens.shape[1]}"
    )


if __name__ == "__main__":
    print("Running backbone tests...")
    test_output_shape_default()
    print("  [PASS] test_output_shape_default")
    test_output_shape_custom_size()
    print("  [PASS] test_output_shape_custom_size")
    test_frozen_by_default()
    print("  [PASS] test_frozen_by_default")
    test_unfreeze_last_2_blocks()
    print("  [PASS] test_unfreeze_last_2_blocks")
    test_grid_size()
    print("  [PASS] test_grid_size")
    test_embed_dim()
    print("  [PASS] test_embed_dim")
    test_no_cls_token_in_output()
    print("  [PASS] test_no_cls_token_in_output")
    print("\nAll backbone tests passed.")
