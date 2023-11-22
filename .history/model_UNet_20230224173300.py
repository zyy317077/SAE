import torch
import torch.nn as nn
import torch.nn.functional as F

# from Deform import DeformConv2d

# class ChannelAttentionModule(nn.Module):
#     def __init__(self, channel, ratio=16):
#         super(ChannelAttentionModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.shared_MLP = nn.Sequential(
#             nn.Conv2d(channel, channel // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // ratio, channel, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avgout = self.shared_MLP(self.avg_pool(x))
#         maxout = self.shared_MLP(self.max_pool(x))
#         return self.sigmoid(avgout + maxout)

# class SpatialAttentionModule(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.sigmoid(self.conv2d(out))
#         return out

# class CBAM(nn.Module):
#     def __init__(self, channel):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttentionModule(channel)
#         self.spatial_attention = SpatialAttentionModule()

#     def forward(self, x):
#         out = self.channel_attention(x) * x
#         out = self.spatial_attention(out) * out
#         return out
class DoubleC(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleC(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:

            #scale_factor指定输出大小为输入的多少倍数，mode:可使用的上采样算法,align_corners为True，输入的角像素将与输出张量对齐，因此将保存下来这些像素的值
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            #nn.ConvTranspose2d是反卷积，对卷积层进行上采样，使其回到原始图片的分辨率
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
# class Up_(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels,bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:

#             #scale_factor指定输出大小为输入的多少倍数，mode:可使用的上采样算法,align_corners为True，输入的角像素将与输出张量对齐，因此将保存下来这些像素的值
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             #nn.ConvTranspose2d是反卷积，对卷积层进行上采样，使其回到原始图片的分辨率
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
    # def forward(self, x1, x2):
    #     x1 = self.up(x1)
    #     # input is CHW
    #     diffY = x2.size()[2] - x1.size()[2]
    #     diffX = x2.size()[3] - x1.size()[3]

    #     x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    #                     diffY // 2, diffY - diffY // 2])
    #     x = torch.cat([x2, x1], dim=1)
    #     # return self.conv(x)
    #     return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.outc = nn.Upsample(size=(360,360),mode='bilinear')
    def forward(self, x):
        # return self.conv(x)
        return self.outc(self.conv(x))

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        self.inc = nn.Upsample(size=(256,256),mode='bilinear')
        # self.inc1 = Conv1(n_channels,64)
        # self.inc2 = Conv2(64,64)
        self.down1 = Down(3, 16)
        # self.inc1 = nn.Conv2d(3,16,3,stride = 2,padding=1)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        factor = 2 if bilinear else 1
        self.down4 = Down(64, 128)
        self.down5 = Down(128, 128)
        self.out = nn.Conv2d(128,3,8,padding=0)

        # self.cbam1 = CBAM(64)
        # self.cbam2 = CBAM(128)
        # self.cbam3 = CBAM(256)
        # self.cbam4 = CBAM(512)

        self.up1 = Up(256, 64, bilinear)
        self.up2 = Up(128, 32, bilinear)
        self.up3 = Up(64, 16, bilinear)
        self.up4 = Up(32, 32, bilinear)
        # self.out1 = Conv3(128,64)
        # self.out2 = Conv4(64,64)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)              # [1,3,256,256]
        # x1 = self.cbam1(x1) + x1
        # x1 = self.inc1(x1)            
        # x1 = self.inc2(x0)
        x2 = self.down1(x1)      # [1,16,128,128]     
        # x2 = self.cbam2(x2) + x2
        x3 = self.down2(x2)      # [1,32,64,64]    
        # x3 = self.cbam3(x3) + x3
        x4 = self.down3(x3)      # [1,64,32,32]
        # x4 = self.cbam4(x4) + x4
        x5 = self.down4(x4)      # [1,128,16,16]    
        x6 = self.down5(x5)      # [1,128,8,8]
        x7 = self.out(x6)        # [1,3,1,1]
   
        x = self.up1(x6, x5)     # [1,64,16,16]
        x = self.up2(x, x4)      # [1,32,32,32]
        x = self.up3(x, x3)      # [1,16,64,64]
        x = self.up4(x, x2)      # [1,8,128,128]
        # x = self.out1(x)
        # x = self.out2(x)
        # x = x0 * x
        logits = self.outc(x)
        # annotation_pred = nn.Softmax(logits)
        return x7,logits

        # out=nn.Softmax(logits)
        # return out
# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.randn(1,3,512,512)
#     unet = UNet(3,19)
#     out1,out2,out3 = unet(x)
#     print(out3)
# if __name__=="__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     unet=UNet(3,3)
#     x=torch.randn([1,3,256,256])
#     out1,out=unet(x)
#     # out = out.to(device)
#     print(out.shape)
#     print(out1.shape)

#     # print(unet)

#     pass;

# model_features = list(unet.children())
# print(model_features[0][3])   #取第0层Sequential()中的第四层
