""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        print('input: {}'.format(x1.shape))
        x2 = self.down1(x1)
        print('input: {}'.format(x2.shape))
        x3 = self.down2(x2)
        print('input: {}'.format(x3.shape))
        x4 = self.down3(x3)
        print('input: {}'.format(x4.shape))
        x5 = self.down4(x4)
        print('input: {}'.format(x5.shape))
        x = self.up1(x5, x4)
        print('input: {}'.format(x.shape))
        x = self.up2(x, x3)
        print('input: {}'.format(x.shape))
        x = self.up3(x, x2)
        print('input: {}'.format(x.shape))
        x = self.up4(x, x1)
        print('input: {}'.format(x.shape))
        logits = self.outc(x)
        print('input: {}'.format(logits.shape))
        return logits
