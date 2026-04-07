<div id="top" align="center">

# DAS-Net: A Lightweight Dynamic Convolution Network with Attention Gates and Deep Supervision for UAV Semantic Segmentation

Young Jae Kim and Sang-Chul Kim

Kookmin University

<a href="#license">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"/>
</a>

</div>

## Overview

DAS-Net extends [ThinDyUNet](https://github.com/SCKIMOSU/uav) with three architectural improvements for UAV semantic segmentation:

1. **Symmetric Dynamic Convolution** — DyConvBlock in both encoder and decoder
2. **Attention Gates** — filter skip connections to suppress irrelevant features
3. **Deep Supervision** — auxiliary loss heads at last 3 decoder stages (λ=0.4)

<img src="./img/Figure3.png" width="100%" alt="DAS-Net Architecture" align=center />

## Results

### Test Set Performance (20K training samples, full test set ~168K images)

| Model | Params (M) | Precision | Recall | Dice | mIoU | ms | FPS |
|-------|-----------|-----------|--------|------|------|----|-----|
| ThinDyUNet | 1.34 | 0.9597 | 0.5937 | 0.6407 | 0.5731 | 8.72 | 114.7 |
| PSPNet | 21.4 | 0.8553 | 0.6686 | 0.6768 | 0.6094 | 13.80 | 72.5 |
| PAN | 21.4 | 0.9055 | 0.7047 | 0.7045 | 0.6438 | 14.77 | 67.7 |
| DeepSupDyUNet | 1.34 | 0.9364 | 0.6838 | 0.7084 | 0.6485 | 8.67 | 115.3 |
| FullDyUNet | 1.63 | 0.8794 | 0.7301 | 0.7394 | 0.6706 | 9.49 | 105.4 |
| UNet | 24.4 | 0.9003 | 0.7333 | 0.7413 | 0.6760 | 14.16 | 70.6 |
| **DAS-Net (Ours)** | **1.66** | **0.8408** | **0.7700** | **0.7506** | **0.6786** | **9.41** | **106.3** |

### Ablation Study

| Model | Sym. Decoder | Attn Gate | Deep Sup | mIoU | Improvement |
|-------|:-:|:-:|:-:|------|-------------|
| ThinDyUNet (baseline) | — | — | — | 0.5731 | — |
| FullDyUNet | ✓ | — | — | 0.6706 | +17.0% |
| DeepSupDyUNet | — | — | ✓ | 0.6485 | +13.2% |
| **DAS-Net (Ours)** | **✓** | **✓** | **✓** | **0.6786** | **+18.4%** |

## Dataset

We use the UAV semantic segmentation dataset proposed by [Kim and Jang (Appl. Sci. 2025)](https://github.com/SCKIMOSU/uav).

- 605,045 paired visible light and infrared images
- Binary segmentation masks (UAV vs. background)
- Training: 20,000 randomly sampled (seed=42)
- Validation: 1,000
- Test: full 168,143 images

## Model Architecture

| Component | Details |
|-----------|---------|
| Encoder | 7-stage DyConvBlock + MaxPool |
| Decoder | 7-stage DyConvBlock + Bilinear Upsample |
| Attention Gate | At each skip connection |
| Deep Supervision | Last 3 decoder stages (λ=0.4) |
| Channels | 64 (fixed) |
| DyConvBlock | K=2 kernels, τ=30, GroupNorm(8) + LeakyReLU |
| Parameters | 1.66M |

## Requirements

```
torch
torchvision
torchinfo
omegaconf
segmentation_models_pytorch
einops
tqdm
```

## How to Run

### Training
```bash
python train_dasnet.py          # DAS-Net (proposed)
python train_thindyunet.py      # ThinDyUNet (baseline)
python train_fulldyunet.py      # FullDyUNet (ablation)
python train_deepsupdyunet.py   # DeepSupDyUNet (ablation)
python train_unet.py            # UNet
python train_pan.py             # PAN
python train_pspnet.py          # PSPNet
```

### Testing
```bash
python test_dasnet.py
python test_thindyunet.py
python test_fulldyunet.py
python test_deepsupdyunet.py
python test_unet.py
python test_pan.py
python test_pspnet.py
```

### Configuration

Each model has a corresponding config file (`config_*.yaml`) with dataset path, hyperparameters, and checkpoint settings.

## Training Details

- Input: 512×512
- Optimizer: AdamW, LR: 1e-4
- Loss: DiceLoss
- Batch size: 24
- Epochs: 50 (early stopping, patience=30)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=15)
- GPU: NVIDIA A6000
- Threshold: 0.5

## Citation

```bibtex
@article{kim2025dasnet,
  title={DAS-Net: A Lightweight Dynamic Convolution Network with Attention Gates and Deep Supervision for UAV Semantic Segmentation},
  author={Kim, Young Jae and Kim, Sang-Chul},
  journal={Applied Sciences},
  year={2025}
}
```



## License

MIT License
