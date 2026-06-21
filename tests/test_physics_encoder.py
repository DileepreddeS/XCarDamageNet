"""Tests for PhysicsTokenEncoder — Stage 1 of XCarDamageNet.

Verifies: output shape (B, N, 396), each head shape, L2-norm of normals,
material probabilities sum to 1, curvature in [0,1], gradient flow.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from xcardamagenet.models.physics_encoder import (
    PhysicsTokenEncoder,
    SurfaceNormalHead,
    MaterialHead,
    ReflectanceHead,
    CurvatureHead,
)


B, N, D = 2, 1369, 384  # batch=2, 37x37 patches, DINOv2 embed_dim


def make_tokens():
    return torch.randn(B, N, D)


# --- Individual head tests ---

def test_normal_head_shape():
    head = SurfaceNormalHead(D)
    out = head(make_tokens())
    assert out.shape == (B, N, 3), f"Expected (B, N, 3), got {out.shape}"


def test_normal_head_unit_vectors():
    """Normals must be L2-normalised (norm≈1 for each token)."""
    head = SurfaceNormalHead(D)
    out = head(make_tokens())
    norms = out.norm(dim=-1)  # (B, N)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"Normal vectors not unit length; max error: {(norms - 1).abs().max():.6f}"
    )


def test_material_head_shape():
    head = MaterialHead(D)
    out = head(make_tokens())
    assert out.shape == (B, N, 6), f"Expected (B, N, 6), got {out.shape}"


def test_material_head_probabilities():
    """Material outputs must be valid probability distributions (sum to 1)."""
    head = MaterialHead(D)
    out = head(make_tokens())
    sums = out.sum(dim=-1)  # (B, N)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"Material probs don't sum to 1; max error: {(sums - 1).abs().max():.6f}"
    )


def test_material_head_non_negative():
    """Material outputs must be non-negative (softmax guarantee)."""
    head = MaterialHead(D)
    out = head(make_tokens())
    assert (out >= 0).all(), "Material probs contain negative values"


def test_reflectance_head_shape():
    head = ReflectanceHead(D)
    out = head(make_tokens())
    assert out.shape == (B, N, 2), f"Expected (B, N, 2), got {out.shape}"


def test_curvature_head_shape():
    head = CurvatureHead(D)
    out = head(make_tokens())
    assert out.shape == (B, N, 1), f"Expected (B, N, 1), got {out.shape}"


def test_curvature_head_range():
    """Curvature must be in [0, 1] (sigmoid guarantee)."""
    head = CurvatureHead(D)
    out = head(make_tokens())
    assert (out >= 0).all() and (out <= 1).all(), (
        f"Curvature out of [0,1]: min={out.min():.4f}, max={out.max():.4f}"
    )


# --- Full encoder tests ---

def test_encoder_output_shape():
    """Full encoder: (B, N, 384) → (B, N, 396)."""
    encoder = PhysicsTokenEncoder(in_dim=D)
    tokens = make_tokens()
    augmented, _ = encoder(tokens)
    assert augmented.shape == (B, N, 396), (
        f"Expected (B, N, 396), got {augmented.shape}"
    )


def test_encoder_output_dim_constant():
    assert PhysicsTokenEncoder.OUTPUT_DIM == 396


def test_encoder_physics_dict_keys():
    """physics_dict must contain all four head outputs."""
    encoder = PhysicsTokenEncoder(in_dim=D)
    _, physics = encoder(make_tokens())
    assert set(physics.keys()) == {"normal", "material", "reflectance", "curvature"}


def test_encoder_physics_dict_shapes():
    encoder = PhysicsTokenEncoder(in_dim=D)
    _, physics = encoder(make_tokens())
    assert physics["normal"].shape == (B, N, 3)
    assert physics["material"].shape == (B, N, 6)
    assert physics["reflectance"].shape == (B, N, 2)
    assert physics["curvature"].shape == (B, N, 1)


def test_encoder_preserves_original_tokens():
    """First 384 dims of output should match input tokens exactly."""
    encoder = PhysicsTokenEncoder(in_dim=D)
    tokens = make_tokens()
    augmented, _ = encoder(tokens)
    assert torch.allclose(augmented[:, :, :384], tokens), (
        "First 384 dims of augmented tokens don't match original input"
    )


def test_encoder_gradient_flow():
    """Gradients must flow back through all four heads."""
    encoder = PhysicsTokenEncoder(in_dim=D)
    tokens = make_tokens().requires_grad_(True)
    augmented, physics = encoder(tokens)
    loss = augmented.sum() + sum(v.sum() for v in physics.values())
    loss.backward()
    assert tokens.grad is not None, "No gradient flowed to input tokens"
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_encoder_parameter_count():
    """Four 2-layer MLPs (hidden=96) give ~149K params for this in_dim=384."""
    encoder = PhysicsTokenEncoder(in_dim=D)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  PhysicsTokenEncoder params: {n_params:,}")
    # 4 × (384*96 + 96 + 96*out + out) + reflectance 4→2 reduce layer ≈ 149K
    assert 100_000 < n_params < 300_000, (
        f"Unexpected param count, got {n_params:,}"
    )


if __name__ == "__main__":
    print("Running PhysicsTokenEncoder tests...")
    test_normal_head_shape();           print("  [PASS] test_normal_head_shape")
    test_normal_head_unit_vectors();    print("  [PASS] test_normal_head_unit_vectors")
    test_material_head_shape();         print("  [PASS] test_material_head_shape")
    test_material_head_probabilities(); print("  [PASS] test_material_head_probabilities")
    test_material_head_non_negative();  print("  [PASS] test_material_head_non_negative")
    test_reflectance_head_shape();      print("  [PASS] test_reflectance_head_shape")
    test_curvature_head_shape();        print("  [PASS] test_curvature_head_shape")
    test_curvature_head_range();        print("  [PASS] test_curvature_head_range")
    test_encoder_output_shape();        print("  [PASS] test_encoder_output_shape")
    test_encoder_output_dim_constant(); print("  [PASS] test_encoder_output_dim_constant")
    test_encoder_physics_dict_keys();   print("  [PASS] test_encoder_physics_dict_keys")
    test_encoder_physics_dict_shapes(); print("  [PASS] test_encoder_physics_dict_shapes")
    test_encoder_preserves_original_tokens(); print("  [PASS] test_encoder_preserves_original_tokens")
    test_encoder_gradient_flow();       print("  [PASS] test_encoder_gradient_flow")
    test_encoder_parameter_count();     print("  [PASS] test_encoder_parameter_count")
    print("\nAll PhysicsTokenEncoder tests passed.")
