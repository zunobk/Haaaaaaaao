# init.py

import os
import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from model import UNetInpainting, PatchGANDiscriminator, weights_init
from dataset import InpaintingDataset, CustomTransforms

def find_all_files(root_dir, extension='*.jpg'):
    return [y for x in os.walk(root_dir) for y in glob.glob(os.path.join(x[0], extension))]


def initialize_model(device):
    generator = UNetInpainting(in_channels=4, out_channels=3).to(device)
    discriminator = PatchGANDiscriminator(in_channels=3).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    return generator, discriminator


def get_dataloader(image_dir, batch_size=8):
    image_paths = find_all_files(image_dir, extension='*.jpg')
    if len(image_paths) == 0:
        raise ValueError("이미지 경로가 비어있습니다. 경로를 확인해주세요.")

    transform = CustomTransforms(train=True)
    dataset = InpaintingDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader


def get_optimizers(generator, discriminator):
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-5)
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.5)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.5)
    return optimizer_g, optimizer_d, scheduler_g, scheduler_d
