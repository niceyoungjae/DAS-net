# 🛠 Installation Guide

This document explains how to set up DAS-Net for training and inference.

---

## System Requirements

### Minimum (Inference Only)
- **OS**: Ubuntu 20.04 / Windows 10+ / macOS 12+
- **Python**: 3.8 – 3.11
- **RAM**: 8 GB
- **Disk**: 5 GB free

### Recommended (Training)
- **GPU**: NVIDIA GPU with ≥ 16 GB VRAM (RTX 3090, A6000, A100)
- **CUDA**: 11.8 or 12.x
- **RAM**: 32 GB
- **Disk**: 50 GB free (for dataset + checkpoints)

### Edge Deployment (Jetson)
- **Device**: NVIDIA Jetson AGX Orin (32 GB)
- **JetPack**: 5.1.1 (L4T R35.3.1)
- **CUDA**: 11.4
- **cuDNN**: 8.6.0
- See [`edge_inference/README.md`](../edge_inference/README.md) for Jetson-specific setup.

---

## Step 1: Clone Repository

```bash
git clone https://github.com/niceyoungjae/DAS-net.git
cd DAS-net
```

---

## Step 2: Create Python Environment

### Option A: Conda (Recommended)
```bash
conda create -n dasnet python=3.10 -y
conda activate dasnet
```

### Option B: venv
```bash
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows
```

---

## Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip wheel

# Install PyTorch (choose CUDA version)
# CUDA 11.8:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.1.0
CUDA: True
```

---

## Step 4: Download Dataset

The UAV semantic segmentation dataset is from [Kim & Jang, *Appl. Sci.* 2025](https://doi.org/10.3390/app15137183):

```bash
# Clone dataset repository
git clone https://github.com/SCKIMOSU/uav.git data/uav_raw
```

### Expected Dataset Structure
```
data/
├── train/
│   ├── images/      ← 304,677 paired VL+IR images
│   └── masks/       ← Binary segmentation masks
├── val/
│   ├── images/      ← 126,360 images
│   └── masks/
└── test/
    ├── images/      ← 174,008 images (full test set)
    └── masks/
```

### Configure Dataset Path

Edit `configs/config_dasnet.yaml`:
```yaml
dataset:
  root: "data/"          # Path to dataset root
  train_subset: 20000    # Stratified random subset per seed
  val_subset: 1000
  test_full: true        # Use all 174,008 test images
```

---

## Step 5: Verify Installation

Run a quick sanity check:
```bash
# Test model definitions
python -c "
from model.DASNet import DASNet
encoder_cfg = {'projection_dim': 64, 'num_heads': 8, 'feed_forward_dim': 128, 'n_trans': 2, 'mlp_head_units': [128, 64]}
model = DASNet(encoder_cfg=encoder_cfg, in_channels=3, num_classes=1)
import torch
x = torch.randn(1, 3, 512, 512)
y = model(x)
print(f'Output shape: {y[\"main\"].shape if isinstance(y, dict) else y.shape}')
print(f'Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M')
"
```

Expected output:
```
Output shape: torch.Size([1, 1, 512, 512])
Total params: 1.66 M
```

---

## Troubleshooting

### CUDA out of memory
Reduce batch size in config:
```yaml
training:
  batch_size: 12  # default 24
```

### `einops` import error
```bash
pip install --upgrade einops
```

### `segmentation_models_pytorch` ImageNet weights download fails
```bash
# Manual download
mkdir -p ~/.cache/torch/hub/checkpoints/
wget https://download.pytorch.org/models/resnet34-b627a593.pth -P ~/.cache/torch/hub/checkpoints/
```

### PyTorch 2.x deprecation warnings
These are safe to ignore. Code is tested on PyTorch 2.0+.

---

## Next Steps

- 📖 Read [`docs/USAGE.md`](USAGE.md) for training and evaluation
- 🔁 Read [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md) to reproduce paper results
- 🛰 Read [`edge_inference/README.md`](../edge_inference/README.md) for Jetson deployment
