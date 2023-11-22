import torch
from torch import nn
from torchvision import models
import os

# from ..utils import initialize_weights
# from .config import vgg19_bn_path


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(_DecoderBlock, self).__init__()
        middle_channels = in_channels // 2
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
                      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(middle_channels),
                      nn.ReLU(inplace=True),
                  ] * (num_conv_layers - 2)
        layers += [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)

def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        #layers.append(nn.BatchNorm2d(out_filters))

    return layers

class DownConv(nn.Module):
    def __init__(self):
        super(DownConv, self).__init__()

        self.model = nn.Sequential(
            *discriminator_block(512, 256),
            # *discriminator_block(128, 128),
            #*discriminator_block(128, 128),
            nn.Conv2d(256, 3, 8, padding=0)
        )

    def forward(self, img_input):
        return self.model(img_input)

class SegNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(SegNet, self).__init__()
        vgg = models.vgg19_bn()
        if pretrained:
            vgg19_bn_path = './initmodel/vgg19_bn_path.pth'
            vgg.load_state_dict(torch.load(vgg19_bn_path))
        features = list(vgg.features.children())
        self.enc1 = nn.Sequential(*features[0:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:27])
        self.enc4 = nn.Sequential(*features[27:40])
        self.enc5 = nn.Sequential(*features[40:])

        self.dc = DownConv()

        self.dec5 = nn.Sequential(
            *([nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)] +
              [nn.Conv2d(512, 512, kernel_size=3, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True)] * 4)
        )
        self.dec4 = _DecoderBlock(1024, 256, 4)
        self.dec3 = _DecoderBlock(512, 128, 4)
        self.dec2 = _DecoderBlock(256, 64, 2)
        self.dec1 = _DecoderBlock(128, num_classes, 2)
        self.initialize_weights(self.dec5, self.dec4, self.dec3, self.dec2, self.dec1)

    def initialize_weights(*models):
        for model in models:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
    def forward(self, x):
        enc1 = self.enc1(x)#[1,64,256,256] 
        enc2 = self.enc2(enc1)#[1,128,128,128]
        enc3 = self.enc3(enc2)#[1,256,64,64]
        enc4 = self.enc4(enc3)#[1,512,32,32]
        enc5 = self.enc5(enc4)#[1,512,16,16]

        out = self.dc(enc5) 

        dec5 = self.dec5(enc5)#[1,512,32,32]
        dec4 = self.dec4(torch.cat([enc4, dec5], 1))#[1,256,64,64]
        dec3 = self.dec3(torch.cat([enc3, dec4], 1))#[1,128,128,128]
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))#[1,64,256,256] 
        dec1 = self.dec1(torch.cat([enc1, dec2], 1))#[1,19,512,512]
        return out,dec1

# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.randn(1,3,512,512)
#     net = SegNet(19)
#     out1 = net(x)
#     print(out1.shape)