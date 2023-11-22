import torch
import torch.nn as nn
import torch.nn.functional as F

class LHSNet(nn.Module):
    def __init__(self):
        super(LHSNet, self).__init__()
        self.u_net_light = UNet(3, 19)

        self.u_net_heavy = u_net_heavy(6, 3)
    def forward(self, x, y, z, coarse):         #x:[1,3,512,512]  y:[1,3,512,512]  z:[1,1,512,512]
        if coarse== True:
            coarse_image = self.u_net_light(x)
            return coarse_image
        elif coarse== False:
            refined_image = self.u_net_heavy(x, y, z)    #[1,3,512,512]
            return refined_image

class DoubleL(nn.Module):
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

class DoubleLayer(nn.Module):
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
            # nn.PixelUnshuffle(2),
            DoubleL(in_channels, out_channels)
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
            self.conv = DoubleLayer(in_channels, out_channels, in_channels // 2)
        else:
            #nn.ConvTranspose2d是反卷积，对卷积层进行上采样，使其回到原始图片的分辨率
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleLayer(in_channels, out_channels)

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

class OutLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.outc = nn.Upsample(size=(512,512),mode='bilinear',align_corners=True)
    def forward(self, x):
        # return self.conv(x)
        return self.outc(self.conv(x))

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        self.inc = nn.Upsample(size=(256,256),mode='bilinear',align_corners=True)
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
        self.outc = OutLayer(32, n_classes)

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
        x = self.outc(x)
        logits = (torch.tanh(x) + 1) / 2 
        # annotation_pred = nn.Softmax(logits)
        return x7,logits

        # out=nn.Softmax(logits)
        # return out
# class Conv_layer(nn.Module):
    
#     def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, up=False):
#         super(Conv_layer,self).__init__()
#         self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)
#         self.norm = nn.InstanceNorm2d(out_dim)
#         self.activation = nn.GELU()
#         self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.up = up
        
#     def forward(self,x):
#         if self.up == True:
#             x = self.upsample(x)
#         x = self.activation(self.norm(self.conv(x)))
#         return x
class LRBlock(nn.Module):
    def __init__(self, channel1,channel2,kernel_size=3,stride=1,padding=1):
        super(LRBlock, self).__init__()
        self.channel1= channel1
        self.channel2= channel2

        self.convblock = nn.Sequential(
            nn.Conv2d(channel1, channel2, kernel_size, stride, padding),
            nn.LeakyReLU(0.05),
            nn.Conv2d(channel1, channel2, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(channel1, channel2, 3, 1, 1),
        )

        # self.A_att_conv = CA(channel1)
        # self.B_att_conv = CA(channel2)

        self.fuse1 = nn.Conv2d(channel1, channel2, 1, 1, 0)
        self.fuse2 = nn.Conv2d(channel1, channel2, 1, 1, 0)
        self.fuse = nn.Conv2d(channel1, channel2, 1, 1, 0)


    def forward(self, x):
        x1,x2=torch.split(x,[self.channel1,self.channel2],dim=1)

        x1 = self.convblock(x1)

        # A = self.A_att_conv(x1)
        P = torch.cat((x2, x1),dim=1)

        # B = self.B_att_conv(x2)
        Q = torch.cat((x1, x2),dim=1)

        c=torch.cat((self.fuse1(P),self.fuse2(Q)),dim=1)
        out=self.fuse(c)
        return out

class res_bolock(nn.Module):
    
    def __init__(self, channels):
        super(res_bolock, self).__init__()
        self.layer_1 = LRBlock(channels, channels, 3, 1, 1)
        self.layer_2 = LRBlock(channels, channels, 3, 1, 1)
        
    def forward(self, x):       #x:[1,512,16,16]
        y = self.layer_1(x)     #[1,512,16,16]
        y = self.layer_2(y)     #[1,512,16,16]
        return y + x

class CAS(nn.Module):
    # context-aware scaling
    def __init__(self, channels, is_up=False):
        super(CAS, self).__init__()
        
        # context extraction
        self.context = CA(channels)
        
        # downsampling
        self.down = nn.Conv2d(channels, channels, 3, 2, 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.is_up = is_up
        
    def forward(self, x):
        con_attention = self.context(x)  #x:[1,64,256,256] con_attention:[1,64,1,1]
        if self.is_up == True:
            x = self.up(x)
        else:
            x = self.down(x)             #x:[1,64,128,128]
        return x * con_attention.expand_as(x)  #x:[1,64,128,128]   #expand_as(tensor)将张量扩展成参数tensor的大小 
    
    
class CA(nn.Module):
    # context attention
    def __init__(self, channels, reduction=16):
        super(CA, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return y
class u_net_heavy(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(u_net_heavy, self).__init__()       
        # initial
        num = 32       
        # encoder
        self.fea_conv = nn.Sequential(
            nn.Conv2d(dim_in, num, 3, 1, 1),
            nn.Conv2d(num, num, 3, 1, 1)
        )        
        self.encoder_1 = LRBlock(num,num*1, 7, 1, 3)
        self.encoder_2 = LRBlock(num*1, num*1, 3, 1, 1)
        self.down_1 = CAS(num*1)
        self.encoder_3 = LRBlock(num*1, num*2, 3, 1, 1)
        self.encoder_4 = LRBlock(num*2, num*2, 3, 1, 1)
        self.down_2 = CAS(num*2)
        self.encoder_5 = LRBlock(num*2, num*4, 3, 1, 1)
        self.encoder_6 = LRBlock(num*4, num*4, 3, 1, 1)
        self.down_3 = CAS(num*4)
        self.encoder_7 = LRBlock(num*4, num*8, 3, 1, 1)
        self.encoder_8 = LRBlock(num*8, num*8, 3, 1, 1)
        self.down_4 = CAS(num*8)
        self.encoder_9 = LRBlock(num*8, num*16, 3, 1, 1)
        self.encoder_10 = LRBlock(num*16, num*16, 3, 1, 1)
        self.down_5 = CAS(num*16)

        # Middle
        blocks = []
        for _ in range(4):
            block = res_bolock(num*16)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        # sementic
        self.sementic_1 = LRBlock(1+3,num*1, 7, 1, 3)
        self.sementic_2 = LRBlock(num*1, num*1, 3, 1, 1)
        self.trans_1 = CAS(num*1)
        self.sementic_3 = LRBlock(num*2, num*2, 3, 1, 1)
        self.sementic_4 = LRBlock(num*2, num*2, 3, 1, 1)
        self.trans_2 = CAS(num*2)
        self.sementic_5 = LRBlock(num*4, num*4, 3, 1, 1)
        self.sementic_6 = LRBlock(num*4, num*4, 3, 1, 1)
        self.trans_3 = CAS(num*4)
        self.sementic_7 = LRBlock(num*8, num*8, 3, 1, 1)
        self.sementic_8 = LRBlock(num*8, num*8, 3, 1, 1)
        self.trans_4 = CAS(num*8)
        self.sementic_9 = LRBlock(num*16, num*16, 3, 1, 1)
        self.sementic_10 = LRBlock(num*16, num*16, 3, 1, 1)
        self.trans_5 = CAS(num*16)

        # decoder
        self.up_1 = CAS(num*32, True)
        self.decoder_1 = LRBlock(num*32, num*32, 3, 1, 1)
        self.decoder_2 = LRBlock(num*32, num*16, 3, 1, 1)
        self.up_2 = CAS(num*24, True)
        self.decoder_3 = LRBlock(num*24, num*24, 3, 1, 1)
        self.decoder_4 = LRBlock(num*24, num*8, 3, 1, 1)
        self.up_3 = CAS(num*12, True)
        self.decoder_5 = LRBlock(num*12, num*12, 3, 1, 1)
        self.decoder_6 = LRBlock(num*12, num*6, 3, 1, 1)
        self.up_4 = CAS(num*8, True)
        self.decoder_7 = LRBlock(num*8, num*8, 3, 1, 1)
        self.decoder_8 = LRBlock(num*8, num*2, 3, 1, 1)
        self.up_5 = CAS(num*3, True)
        self.decoder_9 = LRBlock(num*3, num*1, 3, 1, 1)
        self.decoder_10 = nn.Conv2d(num, dim_out, 3, 1, 1)

    def encoder(self, x, y):
        x = torch.cat([x, y], 1)         #[1,6,512,512]
        x_1 = self.fea_conv(x)
        x_1 = self.encoder_1(x)          #[1,32,512,512]
        x_2 = self.encoder_2(x_1)        #[1,32,512,512]
        x_2 = self.down_1(x_2)           #[1,32,256,256]
        x_3 = self.encoder_3(x_2)        #[1,64,256,256]
        x_4 = self.encoder_4(x_3)        #[1,64,256,256]
        x_4 = self.down_2(x_4)           #[1,64,128,128]
        x_5 = self.encoder_5(x_4)         #[1,128,128,128]
        x_6 = self.encoder_6(x_5)         #[1,128,128,128]
        x_6 = self.down_3(x_6)            #[1,128,64,64]
        x_7 = self.encoder_7(x_6)         #[1,256,64,64]
        x_8 = self.encoder_8(x_7)         #[1,256,64,64]
        x_8 = self.down_4(x_8)            #[1,256,32,32]
        x_9 = self.encoder_9(x_8)         #[1,512,32,32]
        x_10 = self.encoder_10(x_9)       #[1,512,32,32]
        x_10 = self.down_5(x_10)          #[1,512,16,16]
        x_encoder = x_10                  #[1,512,16,16]
        return x_2, x_4, x_6, x_8, x_10

    def sementic(self, coarse, sementic, x_2, x_4, x_6, x_8):  #coarse:[1,3,512,512] sementic:[1,1,512,512]
        x1 = torch.cat([coarse, sementic], 1)              #[1,4,512,512]
        s_1 = self.sementic_1(x1)                          #[1,32,512,512]
        s_2 = self.sementic_2(s_1)                         #[1,32,512,512]
        s_2 = self.trans_1(s_2)                            #[1,32,256,256]
        x2 = torch.cat([s_2, x_2], 1)                      #[1,64,256,256]
        s_3 = self.sementic_3(x2)                          #[1,64,256,256]
        s_4 = self.sementic_4(s_3)                         #[1,64,256,256]
        s_4 = self.trans_2(s_4)                            #[1,64,128,128]
        x3 = torch.cat([s_4, x_4], 1)                      #[1,128,128,128]
        s_5 = self.sementic_5(x3)                          #[1,128,128,128]
        s_6 = self.sementic_6(s_5)                         #[1,128,128,128]
        s_6 = self.trans_3(s_6)                            #[1,128,64,64]
        x4 = torch.cat([s_6, x_6], 1)                      #[1,256,64,64]
        s_7 = self.sementic_7(x4)                          #[1,256,64,64] 
        s_8 = self.sementic_8(s_7)                         #[1,256,64,64]
        s_8 = self.trans_4(s_8)                            #[1,256,32,32]
        x5 = torch.cat([s_8, x_8], 1)                      #[1,512,32,32]
        s_9 = self.sementic_9(x5)                          #[1,512,32,32]
        s_10 = self.sementic_10(s_9)                       #[1,512,32,32]
        s_10 = self.trans_5(s_10)                          #[1,512,16,16]
        s_encoder = s_10
        return s_2, s_4, s_6, s_8, s_10

    def decoder(self, x, s_2, s_4, s_6, s_8, s_10):
        x_11 = self.up_1(torch.cat([x, s_10], 1))          #[1,1024,32,32]      
        x_12 = self.decoder_1(x_11)                        #[1,1024,32,32]
        x_12 = self.decoder_2(x_12)                        #[1,512,32,32]
        x_13 = self.up_2(torch.cat([x_12, s_8], 1))        #[1,768,64,64]
        x_14 = self.decoder_3(x_13)                        #[1,768,64,64]
        x_14 = self.decoder_4(x_14)                        #[1,256,64,64]
        x_15 = self.up_3(torch.cat([x_14, s_6], 1))        #[1,384,128,128]
        x_16 = self.decoder_5(x_15)                        #[1,384,128,128]
        x_16 = self.decoder_6(x_16)                        #[1,192,128,128]
        x_17 = self.up_4(torch.cat([x_16, s_4], 1))        #[1,256,256,256]
        x_18 = self.decoder_7(x_17)                        #[1,256,256,256]
        x_18 = self.decoder_8(x_18)                        #[1,64,256,256]
        x_19 = self.up_5(torch.cat([x_18, s_2], 1))        #[1,96,512,512]
        x_20 = self.decoder_9(x_19)                        #[1,32,512,512]
        x_20 = self.decoder_10(x_20)                       #[1,3,512,512]
        x_20 = (torch.tanh(x_20) + 1) / 2                  #[1,3,512,512]
        return x_20
        
    def forward(self, rain, coarse, sementic):
        x_2, x_4, x_6, x_8, x_10 = self.encoder(rain, coarse)
        s_2, s_4, s_6, s_8, s_10 = self.sementic(coarse, sementic, x_2, x_4, x_6, x_8)
        x_middle =  self.middle(x_10)                  #[1,512,16,16]
        out = self.decoder(x_middle, s_2, s_4, s_6, s_8, s_10) #[1,3,512,512]
        return out
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x1 = torch.randn(1,3,512,512)
    x2 = torch.randn(1,3,512,512)
    x3 = torch.randn(1,1,512,512)
    u_net_heavy = u_net_heavy(6, 3)
    out = u_net_heavy(x1,x2,x3)
    print(out.shape)

# model_features = list(unet.children())
# print(model_features[0][3])   #取第0层Sequential()中的第四层
