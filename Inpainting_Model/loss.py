# loss.py

import torch
import torch.nn as nn
from torchvision import models
from skimage.metrics import structural_similarity as ssim
import numpy as np

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.tv_loss = TVLoss()  # Total Variation Loss 추가

    def forward(self, y_true, y_pred, mask):
        mask = mask.expand_as(y_true)  # 마스크를 이미지 크기에 맞게 확장
        l1_loss = self.l1_loss(y_pred * mask, y_true * mask)
        perceptual_loss = 0.0
        y_true_features = self.perceptual_loss(y_true)
        y_pred_features = self.perceptual_loss(y_pred)
        for true_feature, pred_feature in zip(y_true_features, y_pred_features):
            perceptual_loss += self.l1_loss(pred_feature, true_feature)
        tv_loss = self.tv_loss(y_pred)
        return l1_loss + 0.7 * perceptual_loss + 0.1 * tv_loss  # 가중치 조정

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

def calculate_ssim(outputs, targets, masks):
    outputs = outputs.cpu().numpy().transpose(0, 2, 3, 1)
    targets = targets.cpu().numpy().transpose(0, 2, 3, 1)
    masks = masks.cpu().numpy().transpose(0, 2, 3, 1)
    ssim_vals = [
        ssim(t * m, o * m, multichannel=True, data_range=1.0, win_size=min(t.shape[0], t.shape[1], 7), channel_axis=2)
        for t, o, m in zip(targets, outputs, masks) if min(t.shape[0], t.shape[1]) >= 7 and m.any()]
    return np.mean(ssim_vals)
