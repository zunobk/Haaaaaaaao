# train.py

import torch
import torch.nn as nn
from tqdm import tqdm
from loss import calculate_ssim, CombinedLoss
from utils import visualize_output

def train_one_epoch(generator, discriminator, dataloader, criterion_combined, criterion_gan, optimizer_g, optimizer_d, device, epoch, num_epochs):
    generator.train()
    discriminator.train()
    running_loss_g = 0.0
    running_loss_d = 0.0
    running_ssim = 0.0

    for i, (inputs, targets, masks) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

        # Discriminator 학습 (매 10번째 배치마다)
        if i % 10 == 0:
            optimizer_d.zero_grad()
            real_outputs = discriminator(targets)
            real_labels = torch.ones_like(real_outputs)
            real_loss = criterion_gan(real_outputs, real_labels)

            generated_images = generator(inputs).detach()
            fake_outputs = discriminator(generated_images)
            fake_labels = torch.zeros_like(fake_outputs)
            fake_loss = criterion_gan(fake_outputs, fake_labels)

            loss_d = (real_loss + fake_loss) / 2
            loss_d.backward()
            optimizer_d.step()

        # Generator 학습
        optimizer_g.zero_grad()
        generated_images = generator(inputs)

        # 마스크된 영역만 출력으로 결합
        if generated_images.size() != targets.size():
            generated_images = nn.functional.interpolate(generated_images, size=targets.size()[2:])
        combined_output = targets * (1 - masks) + generated_images * masks

        fake_outputs = discriminator(combined_output)
        real_labels = torch.ones_like(fake_outputs)
        gan_loss = criterion_gan(fake_outputs, real_labels)

        loss_combined = criterion_combined(targets, combined_output, masks)
        loss_g = (gan_loss + loss_combined * 0.5) / 1.5
        loss_g.backward()
        optimizer_g.step()

        running_loss_g += loss_g.item() * inputs.size(0)
        if i % 10 == 0:
            running_loss_d += loss_d.item() * inputs.size(0)

        # SSIM 계산
        with torch.no_grad():
            ssim_val = calculate_ssim(combined_output, targets, masks)
            running_ssim += ssim_val * inputs.size(0)

    epoch_loss_g = running_loss_g / len(dataloader.dataset)
    epoch_loss_d = running_loss_d / (len(dataloader.dataset) // 10)
    epoch_ssim = running_ssim / len(dataloader.dataset)
    return epoch_loss_g, epoch_loss_d, epoch_ssim

def save_model_and_visualize(generator, dataloader, device, epoch):
    generator.eval()
    with torch.no_grad():
        sample_inputs, sample_targets, sample_masks = next(iter(dataloader))
        sample_inputs, sample_targets, sample_masks = sample_inputs.to(device), sample_targets.to(device), sample_masks.to(device)
        sample_outputs = generator(sample_inputs)
        if sample_outputs.size() != sample_targets.size():
            sample_outputs = nn.functional.interpolate(sample_outputs, size=sample_targets.size()[2:])
        combined_output = sample_targets * (1 - sample_masks) + sample_outputs * sample_masks
        visualize_output(sample_inputs, combined_output, sample_targets, sample_masks, save_path=f'output/Final/epoch_{epoch + 1}.png')
        torch.save(generator.state_dict(), f'output/Final/gan_inpainting_generator_epoch_{epoch + 1}.pth')
    generator.train()
