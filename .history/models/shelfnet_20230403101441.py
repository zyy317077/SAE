import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ShelfBlock import Resnet18
# from modules.bn import InPlaceABNSync
from torch.nn import BatchNorm2d
from ShelfBlock import Decoder, LadderBlock

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            # elif isinstance(module, InPlaceABNSync) or isinstance(module, torch.nn.BatchNorm2d):
            #     nowd_params += list(module.parameters())
        return wd_params, nowd_params


class NetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(NetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=3, bias=False,
                                  padding=1)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            # elif isinstance(module, InPlaceABNSync) or isinstance(module, torch.nn.BatchNorm2d):
            #     nowd_params += list(module.parameters())
        return wd_params, nowd_params

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
            *discriminator_block(256, 128),
            # *discriminator_block(128, 128),
            #*discriminator_block(128, 128),
            nn.Conv2d(128, 3, 8, padding=0)
        )

    def forward(self, img_input):
        return self.model(img_input)

class ShelfNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(ShelfNet, self).__init__()
        self.backbone = Resnet18()

        self.decoder = Decoder(planes=64,layers=3,kernel=3)
        self.ladder = LadderBlock(planes=64,layers=3, kernel=3)
        self.dc = DownConv()
        self.conv_out = NetOutput(64, 64, n_classes)
        self.conv_out16 = NetOutput(128, 64, n_classes)
        self.conv_out32 = NetOutput(256, 64, n_classes)

        self.trans1 = ConvBNReLU(128,64,ks=1,stride=1,padding=0)
        self.trans2 = ConvBNReLU(256, 128, ks=1, stride=1, padding=0)
        self.trans3 = ConvBNReLU(512, 256, ks=1, stride=1, padding=0)
    def forward(self, x, aux = True):
        H, W = x.size()[2:]

        feat8, feat16, feat32 = self.backbone(x)#[1,128,64,64],[1,256,32,32],[1,512,16,16]

        feat8 = self.trans1(feat8)#[1,64,64,64]
        feat16 = self.trans2(feat16)#[1,128,32,32]
        feat32 = self.trans3(feat32)#[1,256,16,16]

        out = self.decoder([feat8, feat16, feat32])

        out2 = self.ladder(out)
        out = self.dc(out2[-3])
        feat_cp8, feat_cp16, feat_cp32 = out2[-1], out2[-2], out2[-3]#[1,64,64,64]#[1,128,32,32]#[1,256,16,16]

        feat_out = self.conv_out(feat_cp8)#[1,19,64,64]
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)#[1,19,512,512]

        if aux:
            feat_out16 = self.conv_out16(feat_cp16)#[1,19,32,32]
            feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)#[1,19,512,512]

            feat_out32 = self.conv_out32(feat_cp32)#[1,19,16,16]
            feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)#[1,19,512,512]

            return feat_out, feat_out16, feat_out32
        else:
            return feat_out

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, LadderBlock) or isinstance(child, NetOutput) or isinstance(child, Decoder)\
                    or isinstance(child, ConvBNReLU):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1,3,512,512)
    net = ShelfNet(19)
    out1,out2,out3 = net(x)
    print(out1.shape)