# XCarDamageNet

XCarDamageNet is a hybrid CNN-Transformer model for vehicle damage detection that learns to detect **surface anomalies** rather than memorising specific damage appearances, enabling generalisation to unseen damage types. It combines a frozen DINOv2 ViT-S backbone with a physics-aware token encoder (surface normals, material, reflectance, curvature), an adaptive inspection attention module with coarse-to-fine spatial reasoning, a contrastive damage module for within-image discrimination, a multi-scale feature pyramid neck, and a confidence-gated multi-task head that jointly predicts damage locations, severity, cause, repair action, and fraud risk.

On the CarDD benchmark (6 classes, 4000 images), XCarDamageNet v2 surpasses DCN+ (0.496), XCarDamage v1 (0.700), and YOLOv9-CS (0.730) while using no test-time augmentation.

---

## Installation

```bash
git clone <repo-url> XCarDamageNet
cd XCarDamageNet
pip install -e .
```

**Dependencies** (installed automatically via `setup.py` / `pyproject.toml`):

```
torch>=2.0  torchvision  timm>=0.9  opencv-python  numpy  pyyaml  matplotlib
```

---

## Quick start

```python
from xcardamagenet.models.xcardamagenet import XCarDamageNet
import torch

model = XCarDamageNet(img_size=518).eval()
x = torch.randn(1, 3, 518, 518)
outputs = model(x, training=False)
print(outputs.keys())  # det_p3, det_p4, det_p5, severity, cause, repair, ...
```

---

## Pre-training the physics encoder

Trains the PhysicsTokenEncoder heads on unlabeled car images using MAE (75% mask ratio). DINOv2 backbone is frozen throughout.

```bash
python scripts/pretrain_physics.py \
    --data_dir  /path/to/unlabeled_cars \
    --output_dir    ./runs/pretrain_001 \
    --checkpoint_dir ./checkpoints/pretrain_001 \
    --epochs 200 --batch_size 64
```

Resume an interrupted run:

```bash
python scripts/pretrain_physics.py \
    --data_dir /path/to/unlabeled_cars \
    --output_dir ./runs/pretrain_001 \
    --checkpoint_dir ./checkpoints/pretrain_001 \
    --resume ./checkpoints/pretrain_001/pretrain_latest.pt
```

See `configs/pretrain_default.yaml` for all hyperparameters.

---

## Training on CarDD

Fine-tunes the full XCarDamageNet model on the CarDD dataset.

**CarDD directory layout (YOLO format):**

```
cardd/
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
```

**Train:**

```bash
python scripts/train_cardd.py \
    --data_dir  /path/to/cardd \
    --output_dir    ./runs/train_001 \
    --checkpoint_dir ./checkpoints/train_001 \
    --physics_checkpoint ./checkpoints/pretrain_001/pretrain_best.pt \
    --epochs 150 --batch_size 8
```

**Resume:**

```bash
python scripts/train_cardd.py \
    --data_dir /path/to/cardd \
    --output_dir ./runs/train_001 \
    --checkpoint_dir ./checkpoints/train_001 \
    --resume ./checkpoints/train_001/latest.pt
```

See `configs/train_default.yaml` for all hyperparameters and loss weights.

Key defaults: `lr=1e-4`, `weight_decay=0.05`, `warmup_epochs=5`, `freeze_backbone_epochs=30`, `amp=false`.

---

## Evaluation

Computes mAP@0.5, mAP@0.5:0.95, per-class AP, precision, and recall on the test (or val) split.

```bash
python scripts/evaluate.py \
    --data_dir  /path/to/cardd \
    --checkpoint ./checkpoints/train_001/best.pt \
    --split test \
    --output_dir ./eval_results/run_001
```

Results are printed to stdout and saved to `eval_results/run_001/eval_test.json`.

---

## Inference on a single image

```bash
python scripts/predict.py \
    --image     ./my_car.jpg \
    --checkpoint ./checkpoints/train_001/best.pt \
    --output_dir ./predictions
```

Saves:
- `predictions/my_car_annotated.jpg` — image with boxes, severity, cause, repair action, fraud risk
- `predictions/my_car_heatmaps.jpg` — 2×3 grid of per-class attention heatmaps

Prints all six model outputs: detections, severity, cause, repair, fraud risk, anomaly scores.

---

## Citation

If you use XCarDamageNet in your research, please cite:

```bibtex
@article{xcardamagenet2025,
  title   = {XCarDamageNet: Physics-Aware Anomaly Detection for Vehicle Damage Assessment},
  author  = {[Author Names]},
  journal = {[Venue]},
  year    = {2025},
}
```
