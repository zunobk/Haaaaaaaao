# model.py

import torch.nn as nn
import torch

class UNetInpainting(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(UNetInpainting, self).__init__()

        def down_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            return nn.Sequential(*layers)

        def up_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True):
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(inplace=True)
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            return nn.Sequential(*layers)

        self.encoder1 = down_block(in_channels, 64, use_bn=False)
        self.encoder2 = down_block(64, 128)
        self.encoder3 = down_block(128, 256)
        self.encoder4 = down_block(256, 512)
        self.encoder5 = down_block(512, 512)
        self.encoder6 = down_block(512, 512)
        self.encoder7 = down_block(512, 512)
        self.encoder8 = down_block(512, 512, use_bn=False)

        self.decoder1 = up_block(512, 512)
        self.decoder2 = up_block(1024, 512)
        self.decoder3 = up_block(1024, 512)
        self.decoder4 = up_block(1024, 512)
        self.decoder5 = up_block(1024, 256)
        self.decoder6 = up_block(512, 128)
        self.decoder7 = up_block(256, 64)
        self.decoder8 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)

        d1 = self.decoder1(e8)
        d2 = self.decoder2(torch.cat([d1, e7], dim=1))
        d3 = self.decoder3(torch.cat([d2, e6], dim=1))
        d4 = self.decoder4(torch.cat([d3, e5], dim=1))
        d5 = self.decoder5(torch.cat([d4, e4], dim=1))
        d6 = self.decoder6(torch.cat([d5, e3], dim=1))
        d7 = self.decoder7(torch.cat([d6, e2], dim=1))
        d8 = self.decoder8(torch.cat([d7, e1], dim=1))

        return d8

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
