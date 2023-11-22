import torch
import torch.nn as nn
from torch.nn import functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_sample(x1)
        # diff in dim 2 and dim 3
        diff2 = torch.tensor([x2.size()[2] - x1.size()[2]])
        diff3 = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diff3 // 2, diff3 - diff3 // 2, diff2 // 2, diff2 - diff2 // 2])
        x = torch.cat([x1, x2], dim=1)
        return self.up(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.conv = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up4 = Up(512 + 512, 256)
        self.up3 = Up(256 + 256, 128)
        self.up2 = Up(128 + 128, 64)
        self.up1 = Up(64 + 64, 64)

        self.out = OutConv(64, n_classes)

    def forward(self, x):
        conv1 = self.conv(x)

        conv2 = self.down1(conv1)
        conv3 = self.down2(conv2)
        conv4 = self.down3(conv3)
        conv5 = self.down4(conv4)

        x = self.up4(conv5, conv4)
        x = self.up3(x, conv3)
        x = self.up2(x, conv2)
        x = self.up1(x, conv1)

        logists = self.out(x)
        return logists

def main():
    unet = UNet(3, 1)
    print(unet)

if __name__ == '__main__':
    main()