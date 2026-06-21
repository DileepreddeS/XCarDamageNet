"""Tests for all XCarDamageNet loss functions.

Critical invariants:
- Shape penalty uses w/h RATIOS only (always [0,1]) — never absolute coords
- CB weights computed correctly from effective number of samples
- All losses produce scalar output and have valid gradients
- Weighted combination matches spec: 7.5/0.5/1.5/0.10/0.05/0.02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import math
from xcardamagenet.losses.shape_aware_ciou import ShapeAwareCIoULoss, _shape_penalty
from xcardamagenet.losses.class_balanced import ClassBalancedBCELoss, compute_effective_weights
from xcardamagenet.losses.attention_loss import AttentionSupervisionLoss
from xcardamagenet.losses.contrastive_loss import ContrastiveTripletLoss
from xcardamagenet.losses.physics_loss import PhysicsConsistencyLoss
from xcardamagenet.losses.combined_loss import CombinedLoss

N = 8   # number of samples/boxes
C = 6   # num classes
D = 128 # feature dim


# =====================================================================
# ShapeAwareCIoU Tests
# =====================================================================

def test_shape_penalty_ratio_range():
    """CRITICAL: shape penalty values must always be in [0, 2] using ratios."""
    for _ in range(100):
        w1 = torch.rand(10).abs() + 0.01
        w2 = torch.rand(10).abs() + 0.01
        h1 = torch.rand(10).abs() + 0.01
        h2 = torch.rand(10).abs() + 0.01
        pen = _shape_penalty(w1, h1, w2, h2)
        assert (pen >= 0).all(), "Penalty contains negative values"
        assert (pen <= 2).all(), f"Penalty > 2: max={pen.max():.4f}"


def test_shape_penalty_identical_boxes():
    """Identical boxes should have zero shape penalty."""
    w = torch.tensor([1.0, 2.0, 0.5])
    h = torch.tensor([1.0, 3.0, 0.3])
    pen = _shape_penalty(w, h, w, h)
    assert torch.allclose(pen, torch.zeros_like(pen), atol=1e-6), (
        f"Identical boxes should have zero penalty, got {pen}"
    )


def test_ciou_loss_scalar_output():
    pred = torch.rand(N, 4)
    pred[:, 2:] += pred[:, :2] + 0.1  # ensure x2>x1, y2>y1
    gt = torch.rand(N, 4)
    gt[:, 2:] += gt[:, :2] + 0.1
    cls = torch.randint(0, 6, (N,))

    loss_fn = ShapeAwareCIoULoss()
    loss = loss_fn(pred, gt, cls)
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"


def test_ciou_loss_gradient():
    loss_fn = ShapeAwareCIoULoss()
    pred = torch.rand(N, 4, requires_grad=True)
    pred_box = torch.stack([
        pred[:, 0],
        pred[:, 1],
        pred[:, 0] + pred[:, 2].abs() + 0.01,
        pred[:, 1] + pred[:, 3].abs() + 0.01,
    ], dim=-1)
    gt = torch.rand(N, 4)
    gt[:, 2:] += gt[:, :2] + 0.1
    cls = torch.randint(0, 6, (N,))
    loss = loss_fn(pred_box, gt, cls)
    loss.backward()
    assert pred.grad is not None


def test_ciou_perfect_boxes():
    """Perfect prediction (pred=gt) should give loss ≈ 0."""
    loss_fn = ShapeAwareCIoULoss()
    gt = torch.rand(N, 4)
    gt[:, 2:] += gt[:, :2] + 0.1
    cls = torch.randint(0, 6, (N,))
    loss = loss_fn(gt.clone(), gt, cls)
    assert loss.item() < 0.01, f"Perfect boxes should have ~0 loss, got {loss.item():.4f}"


def test_shape_penalty_uses_ratios_not_absolutes():
    """Verify that large absolute coordinate differences don't explode penalty."""
    # Large boxes (stride-normalised coords could be e.g. 100s)
    w1 = torch.tensor([100.0])
    h1 = torch.tensor([50.0])
    w2 = torch.tensor([50.0])   # 50% different
    h2 = torch.tensor([25.0])   # 50% different
    pen = _shape_penalty(w1, h1, w2, h2)
    # omega = |100-50|/max(100,50) = 0.5, so (1-exp(-0.5))^4 ≈ 0.059
    assert pen.item() < 0.2, (
        f"Shape penalty too large for scale-invariant ratios: {pen.item():.4f}"
    )


# =====================================================================
# ClassBalancedBCE Tests
# =====================================================================

def test_cb_weights_sum_to_num_classes():
    """Effective weights must sum to num_classes."""
    counts = [1847, 2560, 659, 424, 429, 225]
    weights = compute_effective_weights(counts, beta=0.9999)
    assert abs(weights.sum().item() - len(counts)) < 1e-4, (
        f"Weights sum: {weights.sum().item()}, expected {len(counts)}"
    )


def test_cb_weights_rare_class_heavier():
    """Rarer class (tire_flat=225) must have higher weight than common (scratch=2560)."""
    counts = [1847, 2560, 659, 424, 429, 225]
    weights = compute_effective_weights(counts, beta=0.9999)
    scratch_w = weights[1].item()
    tire_w = weights[5].item()
    assert tire_w > scratch_w, (
        f"tire_flat weight ({tire_w:.3f}) should be > scratch ({scratch_w:.3f})"
    )
    print(f"  CB weights: scratch={scratch_w:.3f}, tire_flat={tire_w:.3f}")


def test_cb_loss_scalar():
    loss_fn = ClassBalancedBCELoss()
    logits = torch.randn(N, C)
    targets = torch.randint(0, 2, (N, C)).float()
    loss = loss_fn(logits, targets)
    assert loss.shape == torch.Size([])


def test_cb_loss_gradient():
    loss_fn = ClassBalancedBCELoss()
    logits = torch.randn(N, C, requires_grad=True)
    targets = torch.randint(0, 2, (N, C)).float()
    loss = loss_fn(logits, targets)
    loss.backward()
    assert logits.grad is not None


# =====================================================================
# AttentionSupervisionLoss Tests
# =====================================================================

def test_attn_loss_scalar():
    loss_fn = AttentionSupervisionLoss()
    attn_maps = torch.sigmoid(torch.randn(2, C, 37, 37))
    gt_boxes = [
        torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
        torch.tensor([[0.2, 0.2, 0.8, 0.8]]),
    ]
    gt_classes = [torch.tensor([0]), torch.tensor([2])]
    loss = loss_fn(attn_maps, gt_boxes, gt_classes)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_attn_loss_empty_boxes():
    """No GT boxes → loss = 0."""
    loss_fn = AttentionSupervisionLoss()
    attn_maps = torch.sigmoid(torch.randn(2, C, 37, 37))
    gt_boxes = [torch.zeros(0, 4), torch.zeros(0, 4)]
    gt_classes = [torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)]
    loss = loss_fn(attn_maps, gt_boxes, gt_classes)
    assert loss.item() == 0.0


def test_attn_loss_gradient():
    loss_fn = AttentionSupervisionLoss()
    raw = torch.randn(2, C, 37, 37, requires_grad=True)
    attn_maps = torch.sigmoid(raw)  # non-leaf, raw is the leaf
    gt_boxes = [torch.tensor([[0.1, 0.1, 0.5, 0.5]])]
    gt_classes = [torch.tensor([1])]
    loss = loss_fn(attn_maps[:1], gt_boxes, gt_classes)
    loss.backward()
    assert raw.grad is not None  # check leaf tensor


# =====================================================================
# ContrastiveTripletLoss Tests
# =====================================================================

def test_triplet_loss_scalar():
    loss_fn = ContrastiveTripletLoss()
    anchor = torch.randn(N, D)
    positive = anchor + 0.1 * torch.randn(N, D)
    negative = anchor + 2.0 * torch.randn(N, D)
    loss = loss_fn(anchor, positive, negative)
    assert loss.shape == torch.Size([])


def test_triplet_loss_well_separated():
    """Perfectly separated (anchor≈positive, negative far) → loss ≈ 0."""
    loss_fn = ContrastiveTripletLoss(margin=1.0)
    anchor = torch.zeros(N, D)
    positive = 0.01 * torch.ones(N, D)   # very close
    negative = 10.0 * torch.ones(N, D)   # very far
    loss = loss_fn(anchor, positive, negative)
    assert loss.item() < 0.01, f"Well-separated triplets: {loss.item():.4f}"


def test_triplet_loss_gradient():
    loss_fn = ContrastiveTripletLoss()
    anchor = torch.randn(N, D, requires_grad=True)
    positive = torch.randn(N, D, requires_grad=True)
    negative = torch.randn(N, D, requires_grad=True)
    loss = loss_fn(anchor, positive, negative)
    loss.backward()
    assert anchor.grad is not None


# =====================================================================
# PhysicsConsistencyLoss Tests
# =====================================================================

def test_physics_loss_scalar():
    loss_fn = PhysicsConsistencyLoss()
    physics_logits = torch.randn(N, C)
    pred_logits = torch.randn(N, C)
    loss = loss_fn(physics_logits, pred_logits)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_physics_loss_consistent():
    """When physics-implied == predicted, loss should be low."""
    loss_fn = PhysicsConsistencyLoss()
    logits = torch.zeros(N, C)
    logits[:, 0] = 10.0  # strongly predicts class 0
    loss_consistent = loss_fn(logits, logits.clone())

    # Inconsistent: physics says class 0, pred says class 5
    pred_logits = torch.zeros(N, C)
    pred_logits[:, 5] = 10.0
    loss_inconsistent = loss_fn(logits, pred_logits)

    assert loss_consistent.item() < loss_inconsistent.item(), (
        f"Consistent loss ({loss_consistent:.4f}) should be < inconsistent ({loss_inconsistent:.4f})"
    )


def test_physics_loss_gradient():
    loss_fn = PhysicsConsistencyLoss()
    pred_logits = torch.randn(N, C, requires_grad=True)
    physics_logits = torch.randn(N, C)
    loss = loss_fn(physics_logits, pred_logits)
    loss.backward()
    assert pred_logits.grad is not None


# =====================================================================
# CombinedLoss Tests
# =====================================================================

def test_combined_loss_weights():
    """Verify loss weights match spec exactly."""
    clf = CombinedLoss()
    assert clf.W_BOX == 7.5
    assert clf.W_CLS == 0.5
    assert clf.W_DFL == 1.5
    assert clf.W_ATTN == 0.10
    assert clf.W_CONTRAST == 0.05
    assert clf.W_PHYSICS == 0.02


def test_combined_loss_runs():
    """Full combined loss runs without error and produces scalar."""
    clf = CombinedLoss()
    pred_boxes = torch.rand(N, 4)
    pred_boxes[:, 2:] += pred_boxes[:, :2] + 0.1
    gt_boxes = torch.rand(N, 4)
    gt_boxes[:, 2:] += gt_boxes[:, :2] + 0.1
    class_ids = torch.randint(0, 6, (N,))
    pred_cls = torch.randn(N, C)
    gt_cls = torch.randint(0, 2, (N, C)).float()

    losses = clf(pred_boxes, gt_boxes, class_ids, pred_cls, gt_cls)
    assert "total" in losses
    assert losses["total"].shape == torch.Size([])
    print(f"  Combined total loss: {losses['total'].item():.4f}")
    for k, v in losses.items():
        print(f"    {k}: {v.item():.4f}")


def test_combined_loss_gradient():
    clf = CombinedLoss()
    pred_boxes = torch.rand(N, 4, requires_grad=True)
    pred_boxes_abs = torch.stack([
        pred_boxes[:, 0],
        pred_boxes[:, 1],
        pred_boxes[:, 0] + pred_boxes[:, 2].abs() + 0.01,
        pred_boxes[:, 1] + pred_boxes[:, 3].abs() + 0.01,
    ], dim=-1)
    gt_boxes = torch.rand(N, 4)
    gt_boxes[:, 2:] += gt_boxes[:, :2] + 0.1
    class_ids = torch.randint(0, 6, (N,))
    pred_cls = torch.randn(N, C, requires_grad=True)
    gt_cls = torch.randint(0, 2, (N, C)).float()

    losses = clf(pred_boxes_abs, gt_boxes, class_ids, pred_cls, gt_cls)
    losses["total"].backward()
    assert pred_boxes.grad is not None or pred_cls.grad is not None


if __name__ == "__main__":
    print("Running loss function tests...")
    test_shape_penalty_ratio_range();         print("  [PASS] test_shape_penalty_ratio_range")
    test_shape_penalty_identical_boxes();     print("  [PASS] test_shape_penalty_identical_boxes")
    test_ciou_loss_scalar_output();           print("  [PASS] test_ciou_loss_scalar_output")
    test_ciou_loss_gradient();                print("  [PASS] test_ciou_loss_gradient")
    test_ciou_perfect_boxes();                print("  [PASS] test_ciou_perfect_boxes")
    test_shape_penalty_uses_ratios_not_absolutes(); print("  [PASS] test_shape_penalty_uses_ratios_not_absolutes")
    test_cb_weights_sum_to_num_classes();     print("  [PASS] test_cb_weights_sum_to_num_classes")
    test_cb_weights_rare_class_heavier();     print("  [PASS] test_cb_weights_rare_class_heavier")
    test_cb_loss_scalar();                    print("  [PASS] test_cb_loss_scalar")
    test_cb_loss_gradient();                  print("  [PASS] test_cb_loss_gradient")
    test_attn_loss_scalar();                  print("  [PASS] test_attn_loss_scalar")
    test_attn_loss_empty_boxes();             print("  [PASS] test_attn_loss_empty_boxes")
    test_attn_loss_gradient();                print("  [PASS] test_attn_loss_gradient")
    test_triplet_loss_scalar();               print("  [PASS] test_triplet_loss_scalar")
    test_triplet_loss_well_separated();       print("  [PASS] test_triplet_loss_well_separated")
    test_triplet_loss_gradient();             print("  [PASS] test_triplet_loss_gradient")
    test_physics_loss_scalar();               print("  [PASS] test_physics_loss_scalar")
    test_physics_loss_consistent();           print("  [PASS] test_physics_loss_consistent")
    test_physics_loss_gradient();             print("  [PASS] test_physics_loss_gradient")
    test_combined_loss_weights();             print("  [PASS] test_combined_loss_weights")
    test_combined_loss_runs();                print("  [PASS] test_combined_loss_runs")
    test_combined_loss_gradient();            print("  [PASS] test_combined_loss_gradient")
    print("\nAll loss tests passed.")
