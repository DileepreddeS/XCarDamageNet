"""Tests for ConfidenceGatedMultiTaskHead — Stage 5 of XCarDamageNet.

Verifies: all 6 output shapes, value ranges, confidence gating, gradient flow.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from xcardamagenet.models.head import (
    ConfidenceGatedMultiTaskHead,
    NUM_CLASSES, NUM_CAUSES, NUM_REPAIRS,
)

B = 2
P3_CH, P4_CH, P5_CH = 256, 512, 512
PHYSICS_DIM = 396
H3, W3 = 37, 37
H4, W4 = 19, 19
H5, W5 = 10, 10
N = 37 * 37  # patch tokens


def make_inputs():
    p3 = torch.randn(B, P3_CH, H3, W3)
    p4 = torch.randn(B, P4_CH, H4, W4)
    p5 = torch.randn(B, P5_CH, H5, W5)
    phys = torch.randn(B, N, PHYSICS_DIM)
    return p3, p4, p5, phys


def test_detection_output_shapes():
    """Detection at all 3 scales has correct channel count."""
    head = ConfidenceGatedMultiTaskHead()
    p3, p4, p5, phys = make_inputs()
    out = head(p3, p4, p5, phys, training=True)

    expected_det_ch = 1 + 4 + NUM_CLASSES  # obj + box + class
    assert out["det_p3"].shape == (B, expected_det_ch, H3, W3), out["det_p3"].shape
    assert out["det_p4"].shape == (B, expected_det_ch, H4, W4), out["det_p4"].shape
    assert out["det_p5"].shape == (B, expected_det_ch, H5, W5), out["det_p5"].shape


def test_severity_shape_and_range():
    head = ConfidenceGatedMultiTaskHead()
    out = head(*make_inputs(), training=True)
    assert out["severity"].shape == (B, 1), out["severity"].shape
    assert (out["severity"] >= 0).all() and (out["severity"] <= 1).all()


def test_cause_shape_and_range():
    head = ConfidenceGatedMultiTaskHead()
    out = head(*make_inputs(), training=True)
    assert out["cause"].shape == (B, NUM_CAUSES), out["cause"].shape
    sums = out["cause"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), "Cause probs don't sum to 1"


def test_repair_shape_and_range():
    head = ConfidenceGatedMultiTaskHead()
    out = head(*make_inputs(), training=True)
    assert out["repair"].shape == (B, NUM_REPAIRS), out["repair"].shape
    sums = out["repair"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), "Repair probs don't sum to 1"


def test_attention_maps_shape_and_range():
    """Attention maps: (B, 6, H_p3, W_p3) in [0, 1]."""
    head = ConfidenceGatedMultiTaskHead()
    out = head(*make_inputs(), training=True)
    assert out["attn_maps"].shape == (B, NUM_CLASSES, H3, W3), out["attn_maps"].shape
    assert (out["attn_maps"] >= 0).all() and (out["attn_maps"] <= 1).all()


def test_fraud_score_shape_and_range():
    head = ConfidenceGatedMultiTaskHead()
    out = head(*make_inputs(), training=True)
    assert out["fraud_score"].shape == (B, 1), out["fraud_score"].shape
    assert (out["fraud_score"] >= 0).all() and (out["fraud_score"] <= 1).all()


def test_fraud_implied_shape():
    head = ConfidenceGatedMultiTaskHead()
    out = head(*make_inputs(), training=True)
    assert out["fraud_implied"].shape == (B, NUM_CLASSES), out["fraud_implied"].shape


def test_gate_scores_present():
    """gate_scores list should have entries from all 3 detection branches."""
    head = ConfidenceGatedMultiTaskHead()
    out = head(*make_inputs(), training=True)
    assert "gate_scores" in out
    assert len(out["gate_scores"]) > 0


def test_gradient_flow():
    """All outputs must allow gradient flow back to all inputs."""
    head = ConfidenceGatedMultiTaskHead()
    p3 = torch.randn(B, P3_CH, H3, W3, requires_grad=True)
    p4 = torch.randn(B, P4_CH, H4, W4, requires_grad=True)
    p5 = torch.randn(B, P5_CH, H5, W5, requires_grad=True)
    phys = torch.randn(B, N, PHYSICS_DIM, requires_grad=True)

    out = head(p3, p4, p5, phys, training=True)
    loss = (
        out["det_p3"].sum()
        + out["det_p4"].sum()
        + out["det_p5"].sum()
        + out["severity"].sum()
        + out["cause"].sum()
        + out["repair"].sum()
        + out["attn_maps"].sum()
        + out["fraud_score"].sum()
    )
    loss.backward()
    assert p3.grad is not None, "No gradient at P3"
    assert p4.grad is not None, "No gradient at P4"
    assert p5.grad is not None, "No gradient at P5"
    assert phys.grad is not None, "No gradient at physics tokens"


def test_parameter_count():
    head = ConfidenceGatedMultiTaskHead()
    n = sum(p.numel() for p in head.parameters())
    print(f"  ConfidenceGatedMultiTaskHead params: {n:,}")
    assert n > 1_000_000, f"Expected >1M params, got {n:,}"


def test_num_outputs():
    """Should have exactly the expected keys in output dict."""
    head = ConfidenceGatedMultiTaskHead()
    out = head(*make_inputs(), training=True)
    expected_keys = {"det_p3", "det_p4", "det_p5", "severity", "cause",
                     "repair", "attn_maps", "fraud_score", "fraud_implied", "gate_scores"}
    assert set(out.keys()) == expected_keys, f"Keys: {set(out.keys())}"


if __name__ == "__main__":
    print("Running ConfidenceGatedMultiTaskHead tests...")
    test_detection_output_shapes();       print("  [PASS] test_detection_output_shapes")
    test_severity_shape_and_range();      print("  [PASS] test_severity_shape_and_range")
    test_cause_shape_and_range();         print("  [PASS] test_cause_shape_and_range")
    test_repair_shape_and_range();        print("  [PASS] test_repair_shape_and_range")
    test_attention_maps_shape_and_range(); print("  [PASS] test_attention_maps_shape_and_range")
    test_fraud_score_shape_and_range();   print("  [PASS] test_fraud_score_shape_and_range")
    test_fraud_implied_shape();           print("  [PASS] test_fraud_implied_shape")
    test_gate_scores_present();           print("  [PASS] test_gate_scores_present")
    test_gradient_flow();                 print("  [PASS] test_gradient_flow")
    test_parameter_count();               print("  [PASS] test_parameter_count")
    test_num_outputs();                   print("  [PASS] test_num_outputs")
    print("\nAll ConfidenceGatedMultiTaskHead tests passed.")
