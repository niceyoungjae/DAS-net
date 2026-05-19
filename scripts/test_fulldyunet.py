import os
import time
import torch
from torchvision import transforms
from omegaconf import OmegaConf
from tqdm import tqdm
from model.FullDyUNet import FullDyUNet
from utils.common import get_dataloaders, seg_precision, seg_recall, seg_miou, dice_coeff

config = OmegaConf.load('config_fulldyunet.yaml')
dataset_cfg = config.dataset
trainer_cfg = config.trainer
model_cfg = config.model
ckpt_dir = config.trainer.checkpoint.save_dir

transform = transforms.Compose([
    transforms.Resize(dataset_cfg.image_size),
    transforms.ToTensor()
])

test_dataloader = get_dataloaders(dataset_cfg, transform, test=True)

model = FullDyUNet(
    in_channels=model_cfg.in_channels,
    start_out_channels=model_cfg.start_out_channels,
    num_class=model_cfg.num_classes,
    size=model_cfg.num_blocks,
    padding=model_cfg.num_padding,
    upsample=model_cfg.is_upsample
)

model_path = f'{ckpt_dir}/{config.model.name}-best-val.pth'
device = torch.device(f'cuda:{trainer_cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, weights_only=True)['model_state_dict'])
model.to(device)
model.eval()

if __name__ == '__main__':
    print('Test model: ', model_path)
    total_precision = total_recall = total_miou = total_dice = total_infr_time = 0.0
    total_batches = len(test_dataloader)

    with torch.no_grad():
        for images, true_masks in tqdm(test_dataloader, desc='Model Testing'):
            start_time = time.time()
            images, true_masks = images.to(device), true_masks.to(device)
            outputs = model(images)
            pred_masks = (outputs.sigmoid() > 0.5).float()
            total_infr_time += time.time() - start_time
            total_precision += seg_precision(pred_masks, true_masks)
            total_recall += seg_recall(pred_masks, true_masks)
            total_miou += seg_miou(pred_masks, true_masks)
            total_dice += dice_coeff(pred_masks, true_masks)
            images, true_masks, pred_masks = images.cpu(), true_masks.cpu(), (pred_masks * 255).cpu()

    print('TEST METRICS: ')
    print(f"Precision: {total_precision/total_batches:.4f} | Recall: {total_recall/total_batches:.4f} | Mean IoU: {total_miou/total_batches:.4f} | Dice Coeff: {total_dice/total_batches:.4f} | Inference Time: {total_infr_time/total_batches*100:.2f} ms")
