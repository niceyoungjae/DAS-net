import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import UAVSegmDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import csv
import random
import segmentation_models_pytorch as smp
from omegaconf import OmegaConf


# ══════════════════════════════════════════════
#  Global seed (DASNET_SEED env var를 모든 random에 적용)
#  run_multiseed.py가 seed마다 다른 값을 주입 → 모델 초기화 / 데이터 셔플 / subset 다 다르게
# ══════════════════════════════════════════════
def set_global_seed(seed=None):
    if seed is None:
        seed = int(os.environ.get('DASNET_SEED', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 완전 결정론적이면 너무 느려져서 benchmark는 켜둠 (속도 우선)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"[Seed] Global seed set to {seed}")
    return seed


# 모듈 import 시점에 자동으로 seed 적용 (train_*.py가 utils.common을 import할 때 발동)
_GLOBAL_SEED = set_global_seed()


# ══════════════════════════════════════════════
#  Compound Loss: BCE + Dice + Focal
# ══════════════════════════════════════════════

class CompoundLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal_loss(pred, target)
        return bce_loss + dice_loss + focal_loss


# ══════════════════════════════════════════════
#  CSV Logger
# ══════════════════════════════════════════════

class CSVLogger:
    def __init__(self, ckpt_dir, model_name):
        os.makedirs(ckpt_dir, exist_ok=True)
        self.path = os.path.join(ckpt_dir, f'{model_name}_metrics.csv')
        self.best_iou = 0.0
        self.best_dice = 0.0
        with open(self.path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'pixel_acc', 'mean_iou', 'dice', 'best_iou', 'best_dice'])

    def log(self, epoch, train_loss, val_loss, pixel_acc, mean_iou, dice):
        if mean_iou > self.best_iou:
            self.best_iou = mean_iou
        if dice > self.best_dice:
            self.best_dice = dice
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f'{train_loss:.6f}', f'{val_loss:.6f}',
                             f'{pixel_acc:.4f}', f'{mean_iou:.4f}', f'{dice:.4f}',
                             f'{self.best_iou:.4f}', f'{self.best_dice:.4f}'])
        print(f'[CSV] Best IoU: {self.best_iou:.4f} | Best Dice: {self.best_dice:.4f}')


# ══════════════════════════════════════════════
#  Loss Functions
# ══════════════════════════════════════════════

def get_loss_function(loss_fn):
    if loss_fn == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    elif loss_fn == 'BCEWithLogitsLoss':
        return torch.nn.BCEWithLogitsLoss()
    elif loss_fn == 'DiceLoss':
        return smp.losses.DiceLoss(mode='binary')
    elif loss_fn == 'IoULoss':
        return smp.losses.JaccardLoss(mode='binary')
    elif loss_fn == 'CompoundLoss':
        return CompoundLoss()
    else:
        raise ValueError(f'Loss function {loss_fn} not supported')


# ══════════════════════════════════════════════
#  Optimizer
# ══════════════════════════════════════════════

def get_optimizer(optimizer, model, lr):
    if optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    else:
        raise ValueError(f'Optimizer {optimizer} not supported')


# ══════════════════════════════════════════════
#  DataLoaders
# ══════════════════════════════════════════════

def get_dataloaders(config, tsfm, tsfm_aug=None, test=False):
    if test:
        test_dataset = UAVSegmDataset(config.root, 2, tsfm, "test")
        print('Test dataset size:', len(test_dataset))
        test_dataloader = DataLoader(
            test_dataset, batch_size=config.batch_size,
            shuffle=False, num_workers=config.num_workers,
            collate_fn=custom_collate_fn
        )
        return test_dataloader

    train_dataset = UAVSegmDataset(config.root, 2, tsfm, "train", max_samples=getattr(config, 'max_samples', None))
    print('Train dataset size:', len(train_dataset))

    val_dataset = UAVSegmDataset(config.root, 2, tsfm, "val", max_samples=getattr(config, 'max_samples_val', None))
    print('Val dataset size:', len(val_dataset))

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=custom_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        collate_fn=custom_collate_fn
    )

    for images, masks in train_dataloader:
        print('Train batch size:', images.size())
        break
    for images, masks in val_dataloader:
        print('Val batch size:', images.size())
        break

    return train_dataloader, val_dataloader


# ══════════════════════════════════════════════
#  Config Writer
# ══════════════════════════════════════════════

def write_config(ckpt_dir, config):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    with open(os.path.join(ckpt_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config, f)
    print('Write config.yaml file to:', ckpt_dir)


# ══════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════

def pixel_accuracy(pred_mask, true_mask):
    correct = (pred_mask == true_mask).sum().item()
    total = true_mask.numel()
    return correct / total


def seg_precision(pred_mask, true_mask):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    tp = (pred_mask & true_mask).float().sum((1, 2, 3))
    fp = (pred_mask & ~true_mask).float().sum((1, 2, 3))
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    return precision.mean().item()


def seg_recall(pred_mask, true_mask):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    tp = (pred_mask & true_mask).float().sum((1, 2, 3))
    fn = (~pred_mask & true_mask).float().sum((1, 2, 3))
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    return recall.mean().item()


def seg_miou(pred_mask, true_mask):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    intersection = (pred_mask & true_mask).float().sum((1, 2, 3))
    union = (pred_mask | true_mask).float().sum((1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def dice_coeff(pred_mask, true_mask):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    intersection = (pred_mask & true_mask).float().sum((1, 2, 3))
    dice = (2. * intersection + 1e-6) / (pred_mask.float().sum((1, 2, 3)) + true_mask.float().sum((1, 2, 3)) + 1e-6)
    return dice.mean().item()


# ══════════════════════════════════════════════
#  Collate
# ══════════════════════════════════════════════

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None
    data, masks = zip(*batch)
    return torch.stack(data), torch.stack(masks)
