# 📖 Usage Guide

This document explains how to train and evaluate DAS-Net.

---

## Quick Start

```bash
# 1. Train DAS-Net
python scripts/train_dasnet.py --config configs/config_dasnet.yaml

# 2. Evaluate the trained model
python scripts/test_dasnet.py --config configs/config_dasnet.yaml
```

---

## Training

### Train Individual Models

Each model has its own dedicated training script:

```bash
# Ablation variants (DyConv family)
python scripts/train_dasnet.py        --config configs/config_dasnet.yaml
python scripts/train_thindyunet.py    --config configs/config_thindyunet.yaml
python scripts/train_fulldyunet.py    --config configs/config_fulldyunet.yaml
python scripts/train_deepsupdyunet.py --config configs/config_deepsupdyunet.yaml

# External baselines
python scripts/train_unet.py       --config configs/config_unet.yaml
python scripts/train_mobileunet.py --config configs/config_mobileunet.yaml
python scripts/train_pan.py        --config configs/config_pan.yaml
python scripts/train_pspnet.py     --config configs/config_pspnet.yaml
```

### Training Hyperparameters (default, paper settings)

```yaml
training:
  optimizer: AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-2
  beta1: 0.9
  beta2: 0.999
  batch_size: 24
  epochs: 50
  scheduler:
    type: ReduceLROnPlateau
    factor: 0.5
    patience: 15
    cooldown: 5
  early_stopping:
    patience: 30
  loss: dice
  threshold: 0.5
  augmentation: none  # Deliberately omitted to isolate architectural contributions
```

### Outputs

- **Checkpoints**: `checkpoints/<model_name>/best.pth`
- **Logs**: `logs/<model_name>_<timestamp>.log`
- **TensorBoard**: `runs/<model_name>/`

### Available Models

| Model | Config file | Description |
|---|---|---|
| `dasnet` | `config_dasnet.yaml` | DAS-Net (Ours, 1.66 M) |
| `thindyunet` | `config_thindyunet.yaml` | ThinDyUNet baseline (1.34 M) |
| `fulldyunet` | `config_fulldyunet.yaml` | Symmetric DyConv only (1.63 M) |
| `deepsupdyunet` | `config_deepsupdyunet.yaml` | Deep supervision only (1.34 M) |
| `unet` | `config_unet.yaml` | UNet w/ ResNet-34 (24.4 M) |
| `mobileunet` | `config_mobileunet.yaml` | MobileNetV2 backbone (6.0 M) |
| `pan` | `config_pan.yaml` | PAN w/ ResNet-34 (21.4 M) |
| `pspnet` | `config_pspnet.yaml` | PSPNet w/ ResNet-34 (21.4 M) |

---

## Evaluation

### Per-Model Evaluation

```bash
python scripts/test_dasnet.py        --config configs/config_dasnet.yaml
python scripts/test_thindyunet.py    --config configs/config_thindyunet.yaml
python scripts/test_fulldyunet.py    --config configs/config_fulldyunet.yaml
python scripts/test_deepsupdyunet.py --config configs/config_deepsupdyunet.yaml
python scripts/test_unet.py          --config configs/config_unet.yaml
python scripts/test_mobileunet.py    --config configs/config_mobileunet.yaml
python scripts/test_pan.py           --config configs/config_pan.yaml
python scripts/test_pspnet.py        --config configs/config_pspnet.yaml
```

### Generic Evaluation
```bash
python scripts/eval.py \
    --model dasnet \
    --checkpoint checkpoints/dasnet/best.pth
```

**Output**: Precision, Recall, Dice, mIoU on test set.

---

## Custom Inference

### Load Pre-trained Model
```python
import torch
from model.DASNet import DASNet

# Initialize model
encoder_cfg = {
    'projection_dim': 64,
    'num_heads': 8,
    'feed_forward_dim': 128,
    'n_trans': 2,
    'mlp_head_units': [128, 64],
}
model = DASNet(encoder_cfg=encoder_cfg, in_channels=3, num_classes=1)

# Load checkpoint
ckpt = torch.load('checkpoints/dasnet/best.pth', map_location='cuda')
model.load_state_dict(ckpt['model_state_dict'])
model.cuda().eval()

# Inference on a single image
from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])

img = Image.open('path/to/uav_image.png').convert('RGB')
x = transform(img).unsqueeze(0).cuda()  # (1, 3, 512, 512)

with torch.no_grad():
    output = model(x)
    pred = torch.sigmoid(output['main'] if isinstance(output, dict) else output)
    mask = (pred > 0.5).float()  # Binary mask
```

### Save Prediction
```python
import numpy as np
from PIL import Image

mask_np = mask.cpu().squeeze().numpy().astype(np.uint8) * 255
Image.fromarray(mask_np).save('prediction.png')
```

---

## Multi-Seed Training (Reproducing Paper Results)

To reproduce the paper's three-seed protocol, train each DyConv-family model three times with different seeds:

```bash
# Example: DAS-Net with 3 seeds
for seed in 42 2024 8888; do
    python scripts/train_dasnet.py --config configs/config_dasnet.yaml --seed $seed
done
```

The seed governs:
1. **Subset sampling** (20,000 train + 1,000 val from full dataset)
2. **Model weight initialization**
3. **DataLoader shuffling**

---

## Tips

### Speed up training
- Use **mixed precision** (FP16) — modify training script to use `torch.cuda.amp`
- Increase batch size if GPU memory allows
- Use `num_workers=8` in DataLoader

### Reduce GPU memory
- Reduce batch size to 12 or 8
- Use gradient checkpointing (PyTorch 2.x supports this natively)

### Monitor training
```bash
tensorboard --logdir runs/
```

---

## Next Steps
- 🛠 [`docs/INSTALLATION.md`](INSTALLATION.md) — Setup and dataset preparation
