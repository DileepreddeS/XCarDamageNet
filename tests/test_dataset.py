"""Tests for CarDDDataset — tests without requiring the actual CarDD dataset.

Creates a minimal synthetic dataset structure to verify loading, label parsing,
and augmentation logic work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
import torch
from xcardamagenet.data.cardd_dataset import CarDDDataset, NUM_CLASSES
from xcardamagenet.data.augmentation import CarDDAugmentation, CopyPasteAugmentation
from xcardamagenet.data.preprocessing import preprocess_image, denormalise_boxes


def create_synthetic_dataset(root: Path, split: str = "train", n_images: int = 5):
    """Create a minimal synthetic CarDD-format dataset for testing."""
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    for i in range(n_images):
        # Create synthetic 100×100 image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i:03d}.jpg"), img)

        # Create YOLO-format label (one box per image)
        with open(lbl_dir / f"img_{i:03d}.txt", "w") as f:
            cls = i % NUM_CLASSES
            f.write(f"{cls} 0.5 0.5 0.3 0.3\n")  # centre x,y = 0.5, w,h = 0.3

    return root


def test_dataset_length():
    """Dataset length should match number of images created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = create_synthetic_dataset(Path(tmpdir), n_images=5)
        ds = CarDDDataset(str(root), split="train", img_size=64)
        assert len(ds) == 5, f"Expected 5, got {len(ds)}"


def test_dataset_image_shape():
    """Images should be (3, img_size, img_size) tensors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = create_synthetic_dataset(Path(tmpdir), n_images=3)
        ds = CarDDDataset(str(root), split="train", img_size=64)
        img, target = ds[0]
        assert img.shape == (3, 64, 64), f"Image shape: {img.shape}"
        assert img.dtype == torch.float32


def test_dataset_label_parsing():
    """Labels should parse to boxes in [x1,y1,x2,y2] format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = create_synthetic_dataset(Path(tmpdir), n_images=3)
        ds = CarDDDataset(str(root), split="train", img_size=64)
        img, target = ds[0]

        boxes = target["boxes"]
        classes = target["classes"]

        assert boxes.shape == (1, 4), f"Expected (1,4) boxes, got {boxes.shape}"
        assert classes.shape == (1,), f"Expected (1,) classes, got {classes.shape}"

        # Verify YOLO centre format was converted to corner format correctly
        # 0.5,0.5,0.3,0.3 → x1=0.35, y1=0.35, x2=0.65, y2=0.65
        assert abs(boxes[0, 0].item() - 0.35) < 0.01
        assert abs(boxes[0, 2].item() - 0.65) < 0.01


def test_dataset_target_dict_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = create_synthetic_dataset(Path(tmpdir), n_images=2)
        ds = CarDDDataset(str(root), split="train", img_size=64)
        _, target = ds[0]
        assert "boxes" in target
        assert "classes" in target
        assert "image_id" in target


def test_dataset_no_label_file():
    """Images without label files should return empty boxes/classes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "images" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir = Path(tmpdir) / "labels" / "train"
        lbl_dir.mkdir(parents=True)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "no_label.jpg"), img)
        # No .txt label file created

        ds = CarDDDataset(str(tmpdir), split="train", img_size=64)
        _, target = ds[0]
        assert target["boxes"].shape[0] == 0
        assert target["classes"].shape[0] == 0


def test_collate_fn():
    """collate_fn should produce batched images and list of targets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = create_synthetic_dataset(Path(tmpdir), n_images=4)
        ds = CarDDDataset(str(root), split="train", img_size=64)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=CarDDDataset.collate_fn)
        images, targets = next(iter(loader))
        assert images.shape == (4, 3, 64, 64)
        assert isinstance(targets, list) and len(targets) == 4


# ==================== Augmentation tests ====================

def test_hflip_boxes():
    """Horizontal flip must mirror box x-coordinates correctly."""
    aug = CarDDAugmentation(img_size=64, hflip_prob=1.0)
    img = torch.randn(3, 64, 64)
    boxes = torch.tensor([[0.2, 0.1, 0.5, 0.9]])
    target = {"boxes": boxes, "classes": torch.tensor([0])}
    _, tgt_flip = aug._hflip(img, target)
    # x1_new = 1 - x2_old = 0.5, x2_new = 1 - x1_old = 0.8
    assert abs(tgt_flip["boxes"][0, 0].item() - 0.5) < 1e-5
    assert abs(tgt_flip["boxes"][0, 2].item() - 0.8) < 1e-5


def test_augmentation_shape_preserved():
    """Augmentation must not change image tensor shape."""
    aug = CarDDAugmentation(img_size=64)
    img = torch.randn(3, 64, 64)
    target = {"boxes": torch.tensor([[0.1, 0.1, 0.5, 0.5]]), "classes": torch.tensor([1])}
    img_aug, _ = aug(img, target)
    assert img_aug.shape == (3, 64, 64)


# ==================== Preprocessing tests ====================

def test_preprocess_numpy():
    """preprocess_image must work with numpy BGR array."""
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    tensor, orig_size = preprocess_image(img, img_size=64)
    assert tensor.shape == (1, 3, 64, 64)
    assert orig_size == (200, 300)


def test_denormalise_boxes():
    """Denormalised boxes should scale to pixel coords."""
    boxes = torch.tensor([[0.1, 0.1, 0.9, 0.9]])
    px = denormalise_boxes(boxes, (100, 200))  # (H=100, W=200)
    assert abs(px[0, 0].item() - 20.0) < 1e-4   # 0.1 * W=200
    assert abs(px[0, 1].item() - 10.0) < 1e-4   # 0.1 * H=100
    assert abs(px[0, 2].item() - 180.0) < 1e-4
    assert abs(px[0, 3].item() - 90.0) < 1e-4


if __name__ == "__main__":
    print("Running CarDDDataset tests...")
    test_dataset_length();              print("  [PASS] test_dataset_length")
    test_dataset_image_shape();         print("  [PASS] test_dataset_image_shape")
    test_dataset_label_parsing();       print("  [PASS] test_dataset_label_parsing")
    test_dataset_target_dict_keys();    print("  [PASS] test_dataset_target_dict_keys")
    test_dataset_no_label_file();       print("  [PASS] test_dataset_no_label_file")
    test_collate_fn();                  print("  [PASS] test_collate_fn")
    test_hflip_boxes();                 print("  [PASS] test_hflip_boxes")
    test_augmentation_shape_preserved(); print("  [PASS] test_augmentation_shape_preserved")
    test_preprocess_numpy();            print("  [PASS] test_preprocess_numpy")
    test_denormalise_boxes();           print("  [PASS] test_denormalise_boxes")
    print("\nAll CarDDDataset tests passed.")
