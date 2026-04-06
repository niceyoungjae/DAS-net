import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
from glob import glob
from PIL import Image
import cv2

class UAVSegmDataset(Dataset):
    def __init__(self, root, n_classes, transform, mode, max_samples=None):
        self.root_input = os.path.join(root, "input")
        self.root_mask = os.path.join(root, "labels")
        self.transforms = transform
        self.n_classes = n_classes
        self.mode = mode
        self.img_type = ['infrared', 'visible']
        self.images, self.masks = self.__get_file_path(self.root_input, self.root_mask, self.mode)

        if max_samples is not None and len(self.images) > max_samples:
            import random
            random.seed(123)
            indices = random.sample(range(len(self.images)), max_samples)
            self.images = [self.images[i] for i in sorted(indices)]
            self.masks = [self.masks[i] for i in sorted(indices)]
            
        print(f"Number of files: {len(self.images)} {len(self.masks)}")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)
            mask = (mask > 0.0).float()

        return image, mask

    def __get_file_path(self, root, root_mask, mode):
        image_list, mask_list = [], []
        i = 0
        
        if mode == 'train':
            root = os.path.join(root, 'train')
            root_mask = os.path.join(root_mask, 'train')
        elif mode == 'val':
            root = os.path.join(root, 'val')
            root_mask = os.path.join(root_mask, 'val')
        elif mode == 'test':
            root = os.path.join(root, 'test')
            root_mask = os.path.join(root_mask, 'test')

        for seq in sorted(os.listdir(root)):
            for img_type in self.img_type:
                image_path = sorted(glob(os.path.join(root, seq, img_type, "*.jpg")))
                mask_path = sorted(glob(os.path.join(root_mask, seq, img_type, "*.png")))

                image_list += image_path
                mask_list += mask_path
            
        print(f"Number of files: {len(image_list)} {len(mask_list)}")
        print('Image list:', image_list[-1])
        print('Mask list:', mask_list[-1])
        mask_lookup = set(['/'.join(x.replace('\\', '/').split('/')[-3:])[:-9] for x in mask_list])
        image_list = [x for x in image_list if '/'.join(x.replace('\\', '/').split('/')[-3:])[:-4] in mask_lookup]
        print(f"Number of files: {len(image_list)} {len(mask_list)}") 
        
        assert len(image_list) == len(mask_list), 'Mismatch total images and masks in the dataset'
        return image_list, mask_list
    
if __name__ == '__main__':
    DATASET_PATH = 'C:/test/combined_uav/'
    NUM_CLASSES = 2

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    dataset = UAVSegmDataset(DATASET_PATH, NUM_CLASSES, transform, "val")
    print('Dataset size:', len(dataset))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print('Dataloader size:', len(dataloader))
    i=0
    for qq, (images, masks) in enumerate(dataloader):
        i+=1
        print(i)
        print(images.shape, masks.shape)
        break