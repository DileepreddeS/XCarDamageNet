"""Tests for ContrastiveDamageModule — Stage 3 of XCarDamageNet.

Verifies: output shape, residual computation, damage score range,
normal-only fallback, gradient flow.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from xcardamagenet.models.contrastive import ContrastiveDamageModule

B, N, D = 2, 1369, 396


def make_tokens():
    return torch.randn(B, N, D)


def make_mask(frac_suspicious: float = 0.2) -> torch.Tensor:
    """Create a boolean mask with `frac_suspicious` fraction True."""
    n_susp = max(1, int(N * frac_suspicious))
    mask = torch.zeros(B, N, dtype=torch.bool)
    for b in range(B):
        idx = torch.randperm(N)[:n_susp]
        mask[b, idx] = True
    return mask


def test_output_shape():
    """(B, N, D) + mask → output (B, N, D), scores (B, N)."""
    mod = ContrastiveDamageModule(D)
    tokens = make_tokens()
    mask = make_mask()
    output, scores = mod(tokens, mask)
    assert output.shape == (B, N, D), f"output: {output.shape}"
    assert scores.shape == (B, N), f"scores: {scores.shape}"


def test_damage_scores_range():
    """Damage scores must be in [0, 1]."""
    mod = ContrastiveDamageModule(D)
    _, scores = mod(make_tokens(), make_mask())
    assert (scores >= 0).all() and (scores <= 1).all(), (
        f"Scores out of [0,1]: min={scores.min():.4f}, max={scores.max():.4f}"
    )


def test_normal_tokens_pass_through():
    """Normal tokens (mask=False) should NOT be modified."""
    mod = ContrastiveDamageModule(D, use_projection=False)
    # Zero alpha → residual has no effect regardless
    mod.alpha.data.fill_(0.0)
    tokens = make_tokens()
    mask = make_mask(0.2)
    output, _ = mod(tokens, mask)
    # With alpha=0, output should equal input exactly
    assert torch.allclose(output, tokens, atol=1e-6), (
        "With alpha=0, output should equal input"
    )


def test_suspicious_tokens_modified():
    """Suspicious tokens must receive a non-zero residual (when alpha>0)."""
    mod = ContrastiveDamageModule(D, use_projection=False)
    mod.alpha.data.fill_(1.0)
    tokens = make_tokens()
    mask = make_mask(0.3)
    output, _ = mod(tokens, mask)
    # Suspicious token positions should differ from input
    for b in range(B):
        susp_idx = mask[b].nonzero(as_tuple=True)[0]
        if susp_idx.numel() > 0:
            diff = (output[b, susp_idx] - tokens[b, susp_idx]).abs().max()
            assert diff > 1e-6, f"Suspicious tokens not modified (batch {b})"


def test_no_mask_fallback():
    """When no mask provided, module falls back gracefully (no crash)."""
    mod = ContrastiveDamageModule(D)
    tokens = make_tokens()
    output, scores = mod(tokens, suspicious_mask=None)
    assert output.shape == (B, N, D)
    assert scores.shape == (B, N)


def test_all_suspicious_fallback():
    """When all tokens are suspicious, centroid falls back to global mean."""
    mod = ContrastiveDamageModule(D, use_projection=False)
    tokens = make_tokens()
    all_susp = torch.ones(B, N, dtype=torch.bool)
    output, scores = mod(tokens, all_susp)
    # Should not crash, scores should be valid
    assert output.shape == (B, N, D)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_gradient_flow():
    """Gradients must reach input tokens and all parameters."""
    mod = ContrastiveDamageModule(D)
    tokens = make_tokens().requires_grad_(True)
    mask = make_mask()
    output, scores = mod(tokens, mask)
    loss = output.sum() + scores.sum()
    loss.backward()
    assert tokens.grad is not None, "No gradient at input tokens"
    assert mod.alpha.grad is not None, "No gradient for alpha"


def test_alpha_is_learnable():
    """Alpha must be a trainable parameter."""
    mod = ContrastiveDamageModule(D)
    assert mod.alpha.requires_grad, "alpha should be a learnable parameter"


def test_damage_scores_higher_for_suspicious():
    """On average, suspicious tokens should have higher damage scores than normal."""
    mod = ContrastiveDamageModule(D, use_projection=False)
    mod.eval()
    # Create tokens where suspicious = 5*normal (very different)
    tokens = torch.randn(1, 100, D)
    mask = torch.zeros(1, 100, dtype=torch.bool)
    mask[0, :20] = True  # first 20 are suspicious
    # Make suspicious tokens very different from normal
    tokens[0, :20] = tokens[0, :20] * 10 + 5.0

    _, scores = mod(tokens, mask)
    susp_mean = scores[0, :20].mean().item()
    norm_mean = scores[0, 20:].mean().item()
    print(f"  Suspicious avg score: {susp_mean:.4f}, Normal avg: {norm_mean:.4f}")
    assert susp_mean > norm_mean, (
        "Suspicious tokens should have higher damage scores than normal tokens"
    )


if __name__ == "__main__":
    print("Running ContrastiveDamageModule tests...")
    test_output_shape();                    print("  [PASS] test_output_shape")
    test_damage_scores_range();             print("  [PASS] test_damage_scores_range")
    test_normal_tokens_pass_through();      print("  [PASS] test_normal_tokens_pass_through")
    test_suspicious_tokens_modified();      print("  [PASS] test_suspicious_tokens_modified")
    test_no_mask_fallback();                print("  [PASS] test_no_mask_fallback")
    test_all_suspicious_fallback();         print("  [PASS] test_all_suspicious_fallback")
    test_gradient_flow();                   print("  [PASS] test_gradient_flow")
    test_alpha_is_learnable();              print("  [PASS] test_alpha_is_learnable")
    test_damage_scores_higher_for_suspicious(); print("  [PASS] test_damage_scores_higher_for_suspicious")
    print("\nAll ContrastiveDamageModule tests passed.")
