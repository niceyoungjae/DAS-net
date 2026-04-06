import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

from utils.common import pixel_accuracy, seg_miou, dice_coeff

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    total_dice = 0.0
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for images, true_masks in tqdm(dataloader, desc='Model Evaluation'):
            images = images.to(device)
            true_masks = true_masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, true_masks)

            # Decode the predictions logits
            pred_masks = outputs.sigmoid()
            pred_masks = (pred_masks > 0.9).float()

            running_loss += loss.item()
            total_acc += pixel_accuracy(pred_masks, true_masks)
            total_miou += seg_miou(pred_masks, true_masks)
            total_dice += dice_coeff(pred_masks, true_masks)
        
        avg_acc = total_acc / total_batches
        avg_miou = total_miou / total_batches
        avg_dice = total_dice / total_batches
        
        print('EVAL METRICS: ')
        print(f"Pixel Accuracy: {avg_acc:.4f} | Mean IoU: {avg_miou:.4f} | Dice Coeff: {avg_dice:.4f}")
            
    
    return running_loss / len(dataloader), avg_acc, avg_miou, avg_dice