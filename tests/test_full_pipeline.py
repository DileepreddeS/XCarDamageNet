"""Tests for XCarDamageNet full model — end-to-end integration.

Tests the complete pipeline from raw image to all 6 output types.
Uses small grid size (img_size=448, not 518) to keep tests fast.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from xcardamagenet.models.xcardamagenet import XCarDamageNet

B = 2
IMG_SIZE = 448  # smaller for speed; 448/14 = 32 patches per side
N = 32 * 32     # 1024 patch tokens
NUM_CLASSES = 6


def make_model():
    return XCarDamageNet(img_size=IMG_SIZE, pretrained_backbone=False)


def make_images():
    return torch.randn(B, 3, IMG_SIZE, IMG_SIZE)


def test_forward_produces_all_keys():
    """Full forward pass must return all expected output keys."""
    model = make_model()
    model.eval()
    with torch.no_grad():
        out = model(make_images())

    expected_keys = {
        "det_p3", "det_p4", "det_p5",
        "severity", "cause", "repair",
        "attn_maps", "fraud_score", "fraud_implied",
        "gate_scores", "anomaly_scores", "damage_scores", "physics",
    }
    assert set(out.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(out.keys())}"
    )


def test_detection_output_shapes():
    """Detection at 3 scales with correct shapes."""
    model = make_model()
    model.eval()
    grid_h, grid_w = model.grid_h, model.grid_w  # 32, 32

    with torch.no_grad():
        out = model(make_images())

    det_ch = 1 + 4 + NUM_CLASSES  # 11
    p3_h, p3_w = grid_h, grid_w                   # 32, 32
    p4_h, p4_w = grid_h // 2, grid_w // 2         # 16, 16 (approx)
    p5_h, p5_w = grid_h // 4, grid_w // 4         # 8, 8 (approx)

    assert out["det_p3"].shape[0] == B and out["det_p3"].shape[1] == det_ch
    assert out["det_p4"].shape[0] == B and out["det_p4"].shape[1] == det_ch
    assert out["det_p5"].shape[0] == B and out["det_p5"].shape[1] == det_ch
    print(f"  det_p3: {out['det_p3'].shape}, det_p4: {out['det_p4'].shape}, det_p5: {out['det_p5'].shape}")


def test_severity_range():
    model = make_model()
    with torch.no_grad():
        out = model(make_images())
    assert out["severity"].shape == (B, 1)
    assert (out["severity"] >= 0).all() and (out["severity"] <= 1).all()


def test_cause_probabilities():
    model = make_model()
    with torch.no_grad():
        out = model(make_images())
    assert out["cause"].shape == (B, 5)
    sums = out["cause"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-4)


def test_repair_probabilities():
    model = make_model()
    with torch.no_grad():
        out = model(make_images())
    assert out["repair"].shape == (B, 5)
    sums = out["repair"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-4)


def test_attention_maps():
    model = make_model()
    with torch.no_grad():
        out = model(make_images())
    attn = out["attn_maps"]
    assert attn.shape[0] == B and attn.shape[1] == NUM_CLASSES
    assert (attn >= 0).all() and (attn <= 1).all()


def test_fraud_score_range():
    model = make_model()
    with torch.no_grad():
        out = model(make_images())
    assert out["fraud_score"].shape == (B, 1)
    assert (out["fraud_score"] >= 0).all() and (out["fraud_score"] <= 1).all()


def test_anomaly_scores_range():
    model = make_model()
    with torch.no_grad():
        out = model(make_images())
    scores = out["anomaly_scores"]
    assert scores.shape == (B, N)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_physics_dict_keys():
    model = make_model()
    with torch.no_grad():
        out = model(make_images())
    assert set(out["physics"].keys()) == {"normal", "material", "reflectance", "curvature"}


def test_gradient_flow():
    """Gradients must flow from total loss back to input pixels."""
    model = make_model()
    images = make_images().requires_grad_(True)
    out = model(images, training=True)

    loss = (
        out["det_p3"].sum()
        + out["det_p4"].sum()
        + out["det_p5"].sum()
        + out["severity"].sum()
        + out["cause"].sum()
        + out["attn_maps"].sum()
        + out["fraud_score"].sum()
    )
    loss.backward()
    assert images.grad is not None, "Gradients did not reach input pixels"
    assert not images.grad.isnan().any(), "NaN gradients detected"


def test_parameter_count():
    """Total model should be near 50M params (22M backbone + ~28M novel)."""
    model = make_model()
    counts = model.parameter_count()
    total = counts["total"]
    print(f"  Parameter breakdown:")
    for k, v in counts.items():
        print(f"    {k}: {v:,}")
    # Backbone is 22M, novel components are ~28M → total ~50M
    assert total > 30_000_000, f"Expected >30M params, got {total:,}"
    assert total < 70_000_000, f"Expected <70M params, got {total:,}"


def test_batch_size_1():
    """Model must work with single-image batches."""
    model = make_model()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out = model(x)
    assert out["det_p3"].shape[0] == 1


def test_training_mode_vs_eval():
    """Model should produce valid outputs in both train and eval modes."""
    model = make_model()
    x = make_images()

    model.train()
    out_train = model(x)

    model.eval()
    with torch.no_grad():
        out_eval = model(x)

    # Both should have all keys
    assert set(out_train.keys()) == set(out_eval.keys())


if __name__ == "__main__":
    print("Running XCarDamageNet full pipeline tests...")
    test_forward_produces_all_keys();   print("  [PASS] test_forward_produces_all_keys")
    test_detection_output_shapes();     print("  [PASS] test_detection_output_shapes")
    test_severity_range();              print("  [PASS] test_severity_range")
    test_cause_probabilities();         print("  [PASS] test_cause_probabilities")
    test_repair_probabilities();        print("  [PASS] test_repair_probabilities")
    test_attention_maps();              print("  [PASS] test_attention_maps")
    test_fraud_score_range();           print("  [PASS] test_fraud_score_range")
    test_anomaly_scores_range();        print("  [PASS] test_anomaly_scores_range")
    test_physics_dict_keys();           print("  [PASS] test_physics_dict_keys")
    test_gradient_flow();               print("  [PASS] test_gradient_flow")
    test_parameter_count();             print("  [PASS] test_parameter_count")
    test_batch_size_1();                print("  [PASS] test_batch_size_1")
    test_training_mode_vs_eval();       print("  [PASS] test_training_mode_vs_eval")
    print("\nAll full pipeline tests passed.")
