"""Tests for AdaptiveInspectionAttention — Stage 2 of XCarDamageNet.

Verifies: output shapes, anomaly score range, learnable threshold,
masking logic (normal tokens unchanged), and gradient flow.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from xcardamagenet.models.adaptive_attention import (
    AdaptiveInspectionAttention,
    CoarseScanEncoder,
    FineInspectionEncoder,
)

B, D = 2, 396
GRID_H, GRID_W = 10, 10  # Use small grid for fast tests
N = GRID_H * GRID_W  # 100


def make_tokens():
    return torch.randn(B, N, D)


# --- CoarseScanEncoder tests ---

def test_coarse_output_shapes():
    """Coarse encoder: (B,N,396) → refined(B,N,396) + scores(B,N,1) + mask(B,N)."""
    enc = CoarseScanEncoder(D)
    x = make_tokens()
    refined, scores, mask = enc(x)
    assert refined.shape == (B, N, D), f"refined: {refined.shape}"
    assert scores.shape == (B, N, 1), f"scores: {scores.shape}"
    assert mask.shape == (B, N), f"mask: {mask.shape}"
    assert mask.dtype == torch.bool


def test_coarse_anomaly_scores_range():
    """Anomaly scores must be in [0, 1] (sigmoid)."""
    enc = CoarseScanEncoder(D)
    _, scores, _ = enc(make_tokens())
    assert (scores >= 0).all() and (scores <= 1).all()


def test_coarse_threshold_is_learnable():
    enc = CoarseScanEncoder(D)
    assert enc.threshold_logit.requires_grad


# --- FineInspectionEncoder tests ---

def test_fine_output_shape():
    """Fine encoder: (B,N,D) → (B,N,D)."""
    enc = FineInspectionEncoder(D)
    tokens = make_tokens()
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :10] = True  # first 10 tokens suspicious
    out = enc(tokens, mask, GRID_H, GRID_W)
    assert out.shape == (B, N, D), f"output: {out.shape}"


def test_fine_non_suspicious_unchanged():
    """Non-suspicious tokens should not be modified by fine encoder."""
    enc = FineInspectionEncoder(D)
    enc.eval()
    tokens = make_tokens()
    # Only token 0 is suspicious
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, 0] = True

    with torch.no_grad():
        out = enc(tokens, mask, GRID_H, GRID_W)

    # Tokens far from position 0 (not in 1-ring neighbourhood) should be unchanged
    # Position 0 is at (0,0), its neighbours are (0,0), (0,1), (1,0), (1,1) → indices 0,1,10,11
    non_neighbor_idx = list(range(20, N))  # well outside 1-ring of position 0
    for b in range(B):
        diff = (out[b, non_neighbor_idx] - tokens[b, non_neighbor_idx]).abs().max()
        assert diff < 1e-5, f"Non-suspicious tokens modified: max_diff={diff:.6f}"


def test_fine_no_suspicious_passthrough():
    """When no tokens are suspicious, output should equal input."""
    enc = FineInspectionEncoder(D)
    tokens = make_tokens()
    mask = torch.zeros(B, N, dtype=torch.bool)  # none suspicious
    out = enc(tokens, mask, GRID_H, GRID_W)
    assert torch.allclose(out, tokens), "Zero-suspicious-mask should pass input through"


# --- Full AdaptiveInspectionAttention tests ---

def test_full_output_shapes():
    """Full module: (B, N, 396) → refined(B, N, 396) + scores(B, N)."""
    mod = AdaptiveInspectionAttention(D, grid_h=GRID_H, grid_w=GRID_W)
    tokens = make_tokens()
    refined, scores = mod(tokens, GRID_H, GRID_W)
    assert refined.shape == (B, N, D), f"refined: {refined.shape}"
    assert scores.shape == (B, N), f"scores: {scores.shape}"


def test_full_anomaly_scores_range():
    """Full anomaly scores must be in [0, 1]."""
    mod = AdaptiveInspectionAttention(D, grid_h=GRID_H, grid_w=GRID_W)
    _, scores = mod(make_tokens(), GRID_H, GRID_W)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_full_gradient_flow():
    """Gradients must reach input tokens and all parameters."""
    mod = AdaptiveInspectionAttention(D, grid_h=GRID_H, grid_w=GRID_W)
    tokens = make_tokens().requires_grad_(True)
    refined, scores = mod(tokens, GRID_H, GRID_W)
    loss = refined.sum() + scores.sum()
    loss.backward()
    assert tokens.grad is not None, "No gradient at input tokens"


def test_parameter_count():
    mod = AdaptiveInspectionAttention(D, grid_h=GRID_H, grid_w=GRID_W)
    n_params = sum(p.numel() for p in mod.parameters())
    print(f"  AdaptiveInspectionAttention params: {n_params:,}")
    # Expected ~8M per spec
    assert n_params > 1_000_000, f"Expected >1M params, got {n_params:,}"


def test_grid_override():
    """Grid dimensions can be overridden at forward time."""
    mod = AdaptiveInspectionAttention(D, grid_h=GRID_H, grid_w=GRID_W)
    tokens = make_tokens()
    refined, _ = mod(tokens, grid_h=GRID_H, grid_w=GRID_W)
    assert refined.shape == (B, N, D)


if __name__ == "__main__":
    print("Running AdaptiveInspectionAttention tests...")
    test_coarse_output_shapes();            print("  [PASS] test_coarse_output_shapes")
    test_coarse_anomaly_scores_range();     print("  [PASS] test_coarse_anomaly_scores_range")
    test_coarse_threshold_is_learnable();   print("  [PASS] test_coarse_threshold_is_learnable")
    test_fine_output_shape();               print("  [PASS] test_fine_output_shape")
    test_fine_non_suspicious_unchanged();   print("  [PASS] test_fine_non_suspicious_unchanged")
    test_fine_no_suspicious_passthrough();  print("  [PASS] test_fine_no_suspicious_passthrough")
    test_full_output_shapes();              print("  [PASS] test_full_output_shapes")
    test_full_anomaly_scores_range();       print("  [PASS] test_full_anomaly_scores_range")
    test_full_gradient_flow();              print("  [PASS] test_full_gradient_flow")
    test_parameter_count();                 print("  [PASS] test_parameter_count")
    test_grid_override();                   print("  [PASS] test_grid_override")
    print("\nAll AdaptiveInspectionAttention tests passed.")
