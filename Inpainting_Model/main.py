# main.py

import torch
import matplotlib.pyplot as plt
from init import initialize_model, get_dataloader, get_optimizers
from train import train_one_epoch, save_model_and_visualize
from loss import CombinedLoss
import torch.nn as nn

def main():
    # 데이터셋 경로 설정
    image_dir = 'test2017'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델, 손실 함수 및 옵티마이저 초기화
    generator, discriminator = initialize_model(device)
    dataloader = get_dataloader(image_dir)
    optimizer_g, optimizer_d, scheduler_g, scheduler_d = get_optimizers(generator, discriminator)
    criterion_combined = CombinedLoss().to(device)
    criterion_gan = nn.BCELoss().to(device)

    # 학습 루프
    num_epochs = 2500
    epoch_losses_g = []
    epoch_losses_d = []
    epoch_ssim_vals = []

    for epoch in range(num_epochs):
        epoch_loss_g, epoch_loss_d, epoch_ssim = train_one_epoch(
            generator, discriminator, dataloader, criterion_combined, criterion_gan,
            optimizer_g, optimizer_d, device, epoch, num_epochs)

        epoch_losses_g.append(epoch_loss_g)
        epoch_losses_d.append(epoch_loss_d)
        epoch_ssim_vals.append(epoch_ssim)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Generator Loss: {epoch_loss_g:.4f}, Discriminator Loss: {epoch_loss_d:.4f}, SSIM: {epoch_ssim:.4f}')

        scheduler_g.step()
        scheduler_d.step()

        # 중간 시각화 및 모델 저장
        if (epoch + 1) % 10 == 0:
            save_model_and_visualize(generator, dataloader, device, epoch)

    # 최종 모델 저장
    torch.save(generator.state_dict(), 'gan_inpainting_generator_final.pth')

    # 학습 손실 및 메트릭 시각화
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(range(num_epochs), epoch_losses_g, 'g-', label='Generator Loss')
    ax1.plot(range(num_epochs), epoch_losses_d, 'r-', label='Discriminator Loss')
    ax2.plot(range(num_epochs), epoch_ssim_vals, 'c-', label='SSIM')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='g')
    ax2.set_ylabel('Metric', color='c')

    plt.title('Training Loss and Metrics')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()
