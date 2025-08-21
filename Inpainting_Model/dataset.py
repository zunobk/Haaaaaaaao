# dataset.py

import os
import glob
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomTransforms:
    def __init__(self, train=True):
        self.image_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.train = train

    def __call__(self, image, mask):
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        image = self.image_transforms(image)
        mask = self.mask_transforms(mask)
        return image, mask

def create_irregular_mask(image_shape, max_vertex=8, max_length=80, max_brush_width=20, max_angle=360):
    mask = np.zeros(image_shape[:2], np.uint8)
    num_v = random.randint(1, max_vertex)
    for i in range(num_v):
        start_x = random.randint(0, image_shape[1])
        start_y = random.randint(0, image_shape[0])
        for j in range(1, random.randint(1, 5)):
            angle = random.uniform(0, max_angle)
            length = random.randint(5, max_length)
            brush_width = random.randint(10, max_brush_width)
            end_x = start_x + int(length * np.cos(np.radians(angle)))
            end_y = start_y + int(length * np.sin(np.radians(angle)))
            cv2.line(mask, (start_x, start_y), (end_x, end_y), 1, brush_width)
            start_x, start_y = end_x, end_y
    return mask

class InpaintingDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = create_irregular_mask(image_rgb.shape, max_vertex=8, max_length=80, max_brush_width=20)
        if self.transform:
            image, mask = self.transform(image_rgb, mask)
            mask = (mask > 0).float()
            mask = mask.expand(3, -1, -1)
        masked_image = image * (1 - mask)
        input_image = torch.cat((masked_image, mask[0:1]), dim=0)
        return input_image, image, mask
