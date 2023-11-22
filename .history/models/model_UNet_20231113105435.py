import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.ESPNet import ESPNet
# from models.shelfnet import ShelfNet
# from models.SegNet import SegNet

class LHSNet(nn.Module):
    def __init__(self):
        super(LHSNet, self).__init__()
        self.u_net_light = UNet(3, 19)
        # self.u_net_light = ESPNet(19,2,3,None)
        # self.u_net_light = ShelfNet(19)
        # self.u_net_light = SegNet(19)

        self.u_net_heavy = u_net_heavy(6, 3)
    def forward(self, x, y, z, coarse):         #x:[1,3,512,512]  y:[1,3,512,512]  z:[1,1,512,512]
        if coarse== True:
            coarse_image = self.u_net_light(x)
            return coarse_image
        elif coarse== False:
            refined_image = self.u_net_heavy(x, y, z)    #[1,3,512,512]
            return refined_image

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, activation=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down_scale(nn.Module):
    def __init__(self, in_channels):
        super(Down_scale, self).__init__()
        self.main = BasicConv(in_channels, in_channels, 3, 2)
    def forward(self, x):
        return self.main(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.DWT = DWT()
        self.maxpool_conv = nn.Sequential(
            # nn.MaxPool2d(2),
            # # nn.PixelUnshuffle(2),
            Down_scale(in_channels),
            DoubleConv(in_channels, out_channels)
        )
        # self.maxpool_conv = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     # nn.PixelUnshuffle(2),
        #     DoubleConv(in_channels, out_channels)
        # )
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # self.IDWT = IDWT()
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

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
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
        x = self.outc(x)
        logits = (torch.tanh(x) + 1) / 2 
        # annotation_pred = nn.Softmax(logits)
        return x7,logits

        # out=nn.Softmax(logits)
        # return out
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)
class Conv_layer(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, up=False):
        super(Conv_layer,self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_dim)
        self.activation = nn.GELU()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up = up

    def forward(self,x):
        if self.up == True:
            x = self.upsample(x)
        x = self.activation(self.norm(self.conv(x)))
        return x

class Res2NetConv(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Res2NetConv,self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),   #[1,64,512,512]
            nn.LeakyReLU(inplace=True),
            Res2NetBottleneck(out_dim, out_dim),
        )
        self.down = nn.Conv2d(out_dim, out_dim, 3, 2, 1)

    def forward(self,x): #[1,6,512,512]
        x = self.conv(x) #[1,64,512,512]
        x = self.down(x)
        return x

class Res2NetBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1,sa=True, ca=True, norm_layer=None):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d
        bottleneck_planes = groups * planes
        # self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.conv1 =nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        # self.conv3 = conv1x1(bottleneck_planes, planes * self.expansion)
        self.conv3 =nn.Conv2d(bottleneck_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.LeakyReLU(inplace=True)
        self.sa = SA() if sa else None
        self.ca = CA(planes * self.expansion) if ca else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x         #[1,64,512,512]

        out = self.conv1(x)  #[1,64,512,512]
        out = self.bn1(out)
        out = self.relu(out) #[1,64,512,512]

        xs = torch.chunk(out, self.scales, 1) #[1,16,512,512]
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])       #[1,16,512,512]
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))  #[1,16,512,512]
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1])))) #[1,16,512,512]
        if self.sa is not None:
            ys_0 = self.sa(ys[0])#[1,16,512,512]
            ys_1 = self.sa(ys[1])#[1,16,512,512]
            ys_2 = self.sa(ys[2])#[1,16,512,512]
            ys_3 = self.sa(ys[3])#[1,16,512,512]
        
        # out = torch.cat(ys, 1)  #[1,64,512,512]
        out = torch.cat([ys_0,ys_1,ys_2,ys_3], 1)#[1,64,512,512]

        # out = self.conv3(out)   #[1,64,512,512]
        # out = self.bn3(out)

        if self.ca is not None:
            out = self.ca(out) #[1,64,512,512]

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity        #[1,64,512,512]
        out = self.relu(out)

        return out
class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, residual=True):
        super(ResBlock, self).__init__()

        self.residual = residual

        self.body = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size, stride, padding),
        )

    def forward(self, x):
        identity = x
        x = self.body(x)
        if self.residual:
            out = identity + x
        else:
            out = x
        return out
        
class ERB(nn.Module):
    def __init__(self, n_feats, ratio=2):
        super(ERB, self).__init__()
        self.conv1 = RRRB(n_feats, ratio)
        self.activation = nn.LeakyReLU(0.1)
        self.conv2 = RRRB(n_feats, ratio)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        return out

class RRRB(nn.Module):
    def __init__(self, n_feats, ratio=2):
        super(RRRB, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, ratio*n_feats, 1, 1, 0)
        self.fea_conv = nn.Conv2d(ratio*n_feats, ratio*n_feats, 3, 1, 0)
        self.reduce_conv = nn.Conv2d(ratio*n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        out = self.expand_conv(x)
        out_identity = out
        
        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identity
        out = self.reduce_conv(out)
        out += x

        return out
def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t

class AtrousBlockPad2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_bias, use_scale, activation,
                 needs_projection = False,atrousBlock=[1, 2, 4, 8]):
        super(AtrousBlockPad2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.activation = activation
        self.atrousBlocks = atrousBlock
        # self.needs_projection = self.stride > 1
        self.dims_match = self.in_channels != self.out_channels
        # self.needs_projection = self.needs_projection or self.dims_match
        self.needs_projection = needs_projection

        if self.needs_projection:
            self.projection = Conv2D_ReflectPad(in_channels=self.in_channels,
                                                out_channels=self.out_channels,
                                                kernel_size=1,
                                                stride=self.stride,
                                                use_bias=self.use_bias,
                                                act=self.activation)
        self.atrous_layers = []

        for i in range(4):
            self.atrous_layers.append(AtrousConv2D_ReflectPad(in_channels=self.out_channels,
                                                              out_channels=int(self.out_channels / 4),
                                                              kernel_size=self.kernel_size,
                                                              stride=self.stride,
                                                              dilation=atrousBlock[i],
                                                              use_bias=self.use_bias,
                                                              use_scale=self.use_scale,
                                                              act=self.activation))
        self.atrous_layers = nn.Sequential(*self.atrous_layers)

        self.conv1 = Conv2D_ReflectPad(in_channels=self.out_channels,
                                       out_channels=self.out_channels,
                                       kernel_size=self.kernel_size,
                                       stride=self.stride,
                                       use_bias=self.use_bias,
                                       use_scale=self.use_scale,
                                       act=self.activation
                                       )

    def forward(self, input):
        if self.needs_projection:
            input = self.projection(input)

        x1 = self.atrous_layers[0](input) #[1,128,16,16]
        x2 = self.atrous_layers[1](input) #[1,128,16,16]
        x3 = self.atrous_layers[2](input) #[1,128,16,16]
        x4 = self.atrous_layers[3](input) #[1,128,16,16]

        x = torch.cat((x1, x2, x3, x4), 1)    #[1,512,16,16]
        x5 = self.conv1(x)                #[1,512,16,16]

        return input + x5
class ConvAftermath(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True, use_scale=True, norm=None, act=None):
        super(ConvAftermath, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.norm = norm
        self.act = act
        self.b = None
        self.s = None

    def forward(self, input):
        net = input
        if self.use_bias and self.b is not None:
            net = net + self.use_bias
        if self.use_scale and self.s is not None:
            net = net * self.s
        if self.norm is not None:
            net = self.norm(net)
        if self.act is not None:
            net = self.act(net)
        return net
class Conv2D_ReflectPad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_bias=True, use_scale=True, norm=None,
                 act=None,
                 padding='same', padding_algorithm="reflect"):
        super(Conv2D_ReflectPad, self).__init__()
        self.padding = padding
        self.padding_algorithm = padding_algorithm
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.strides = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.atrous_rate = 1
        if padding == 'same':
            self.padding = self.kernel_size // 2 if self.strides == 1 else 0
            self.pad_flag = True
        else:
            self.padding = padding

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.strides,
                              bias=False,
                              padding=self.padding,
                              padding_mode=self.padding_algorithm)
        self.conv_aftermath = ConvAftermath(in_channels=self.out_channels,
                                            out_channels=self.out_channels,
                                            use_bias=self.use_bias,
                                            use_scale=self.use_scale,
                                            norm=self.norm,
                                            act=self.act)

    def forward(self, input):
        x = self.conv(input)
        y = self.conv_aftermath(x)
        return y

class AtrousConv2D_ReflectPad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, use_bias=True, use_scale=True,
                 norm=None,
                 act=None):
        super(AtrousConv2D_ReflectPad, self).__init__()
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad_flag = False
        self.dilation = dilation
        self.padding = dilation

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              dilation=self.dilation,
                              padding=self.padding,
                              padding_mode="reflect",
                              bias=False)
        self.conv_aftermath = ConvAftermath(in_channels=self.out_channels, out_channels=self.out_channels,
                                            use_bias=self.use_bias,
                                            use_scale=self.use_scale, norm=self.norm, act=self.act)

    def forward(self, input): #[1,512,16,16]
        net = self.conv(input) #[1,256,16,16]
        net = self.conv_aftermath(net) #[1,256,16,16]
        return net


class CAS(nn.Module):
    # context-aware scaling
    def __init__(self, channels,is_up=False):
        super(CAS, self).__init__()
        
        # self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        # self.RB1 = ResBlock(channels,channels)
        # self.RB2 = ResBlock(channels,channels)
        # context extraction
        self.context = CA(channels)      
        # self.context = DAB(channels,16)
        # downsampling
        self.down = nn.Conv2d(channels, channels, 3, 2, 1)
        # upsampling
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.is_up = is_up
        
    def forward(self, x):
        # x = self.conv1(x)                #x:[1,64,512,512]
        # x = self.RB1(x)                  #x:[1,64,512,512]
        # x1 = self.RB2(x)                 #x:[1,64,512,512]
        # x2 = torch.cat([x, x1], dim=1)   #x:[1,128,512,512]
        con_attention = self.context(x)  #x:[1,64,256,256] con_attention:[1,128,512,512]
        if self.is_up == True:
            x = self.up(x)
        else:
            x = self.down(con_attention)  #x:[1,64,256,256]
        # x =  self.conv2(x)         #x:[1,64,256,256]
        return x   #x:[1,64,128,128]   #expand_as(tensor)将张量扩展成参数tensor的大小 

class DAB(nn.Module):
    def __init__(self, channels, reduction=16):
        super(DAB, self).__init__()
        # self.conv = nn.Conv2d(channels, channels,3,stride=1,padding=1)
        # modules_body = []
        # for i in range(2):
        #     modules_body.append(self.conv)
        #     # if bn: modules_body.append(nn.BatchNorm2d(channels))
        #     if i == 0: modules_body.append(nn.ReLU(True))
        
        self.SA = spatial_attn_layer()        ## Spatial Attention
        self.CA = CA(channels, reduction)     ## Channel Attention
        # self.body = nn.Sequential(*modules_body)
        self.conv1x1 = nn.Conv2d(channels+1, channels, kernel_size=1)


    def forward(self, x): #[1,32,512,512]
        # res = self.body(x)  
        res = x               #[1,32,512,512]
        sa_branch = self.SA(res) #[1,1,512,512]
        ca_branch = self.CA(res) #[1,32,512,512]
        res = torch.cat([sa_branch, ca_branch], dim=1) #[1,33,512,512]
        res = self.conv1x1(res)  #[1,32,512,512]
        res += x               #[1,32,512,512]  
        return res    
    
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
        return y * x

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1,padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class SA(nn.Module):
    def __init__(self, kernel_size=3):
        super(SA, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1,kernel_size=3,stride=1,padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x
class PA(nn.Module):
    def __init__(self, n_feats, reduction=8):
        super(PA, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats//reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_feats//reduction, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):   #[1,16,512,512]
        att = self.body(x)  #[1,1,512,512]
        out = att * x       #[1,16,512,512]
        return out

class FFModule(nn.Module):
    def __init__(self,out_channels):
        super(FFModule, self).__init__()

        self.conv = conv3x3(4, 4)
        self.fuse1 = nn.Conv2d(3, out_channels, 1, 1, 0)
        self.fuse2 = nn.Conv2d(1, out_channels, 1, 1, 0)
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0)
        self.conv2 = CA(out_channels)      
        # self.context = DAB(channels,16)
        # downsampling
        self.down = nn.Conv2d(out_channels, out_channels, 3, 2, 1)

    def forward(self, x1,x2):

        x = torch.cat([x1,x2], 1)
        w = F.softmax(self.conv(x), dim=1) #[1,4,512,512]
        x1 = x1 * w[:, 0:3, :, :] #[1,3,512,512]
        x2 = x2 * w[:, 3:4, :, :] #[1,1,512,512]
        x = torch.cat([self.fuse1(x1),self.fuse2(x2)], 1) #[1,64,512,512]
        # x = torch.cat([x1,x2],1)
        x = self.fuse(x) #[1,32,512,512]
        x = self.conv2(x) #[1,32,512,512]
        y = self.down(x) #[1,32,256,256]
        return y

class DeBlock(nn.Module):
    def __init__(self, dim, kernel_size=3,stride=1,padding=1):
        super(DeBlock, self).__init__()

        # self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size,stride, padding, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size,stride, padding, bias=True)
        self.calayer = CA(dim)
        self.palayer = PA(dim)

    def forward(self, x):             #[1,1024,32,32]
        res = self.act1(self.conv1(x))  #[1,1024,32,32]
        res1 = res + x                   #[1,1024,32,32]
        res2 = self.act1(self.conv2(res1)) #[1,1024,32,32]
        res3 = res + res2 
        res4 = self.calayer(res3)         #[1,1024,32,32]
        res5 = self.palayer(res4)         #[1,1024,32,32]
        res = res5 + res2                        #[1,1024,32,32]

        return res

class u_net_heavy(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(u_net_heavy, self).__init__()       
        # initial
        num = 32       
        # encoder
        # self.encoder_2 = Res2NetConv(dim_in,num*1)
        # self.encoder_2 = ResBlock(num*1, num*1, 3, 1, 1)
        self.encoder_1 = nn.Conv2d(dim_in, num*1, 3, 1, 1)
        self.encoder_2 = ResBlock(num*1, num*1, 3, 1, 1)
        self.down_1 = CAS(num*1)
        self.encoder_3 = nn.Conv2d(num*1, num*2, 3, 1, 1)
        self.encoder_4 = ResBlock(num*2, num*2, 3, 1, 1)
        self.down_2 = CAS(num*2)
        self.encoder_5 = nn.Conv2d(num*2, num*4, 3, 1, 1)
        self.encoder_6 = ResBlock(num*4, num*4, 3, 1, 1)
        self.down_3 = CAS(num*4)
        self.encoder_7 = nn.Conv2d(num*4, num*8, 3, 1, 1)
        self.encoder_8 = ResBlock(num*8, num*8, 3, 1, 1)
        self.down_4 = CAS(num*8)
        self.encoder_9 = nn.Conv2d(num*8, num*16, 3, 1, 1)
        self.encoder_10 = ResBlock(num*16, num*16, 3, 1, 1)
        self.down_5 = CAS(num*16)

        # Middle
        blocks = []
        for _ in range(4):
            block = ResBlock(num*16,num*16)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)
        # self.middle = AtrousBlockPad2(in_channels= num*16, out_channels= num*16, kernel_size=3, stride=1, use_bias=True, use_scale=True, activation=nn.ReLU(True))
        # self.middle = AtrousBlockPad2(in_channels= num*16, out_channels= num*16, kernel_size=3, stride=1, use_bias=True, use_scale=True, activation=nn.ReLU(True))

        # sementic
        self.sementic_2 = Res2NetConv(1+3,num*1)
        # self.sementic_2 = ResBlock(num*1, num*1, 3, 1, 1)
        # self.conv = conv3x3(4, 4)
        # self.sementic_2 = Conv_layer(3,num*1, 3, 1, 1)
        # self.trans_1 = CAS(num*1)
        self.ffm = FFModule(num*1)
        # self.sementic_3 = ResBlock(num*2, num*2, 3, 1, 1)
        self.sementic_4 = Res2NetConv(num*2, num*2)
        # self.trans_2 = CAS(num*2)
        # self.sementic_5 = ResBlock(num*4, num*4, 3, 1, 1)
        self.sementic_6 = Res2NetConv(num*4, num*4)
        # self.trans_3 = CAS(num*4)
        # self.sementic_7 = ResBlock(num*8, num*8, 3, 1, 1)
        self.sementic_8 = Res2NetConv(num*8, num*8)
        # self.trans_4 = CAS(num*8)
        # self.sementic_9 = ResBlock(num*16, num*16, 3, 1, 1)
        self.sementic_10 = Res2NetConv(num*16, num*16)
        # self.trans_5 = CAS(num*16)
        # self.sementic_1 = Conv_layer(1+3,num*1, 7, 1, 3)
        # self.sementic_2 = Conv_layer(num*1, num*1, 3, 1, 1)
        # self.trans_1 = CAS(num*1)
        # self.sementic_3 = Conv_layer(num*2, num*2, 3, 1, 1)
        # self.sementic_4 = Conv_layer(num*2, num*2, 3, 1, 1)
        # self.trans_2 = CAS(num*2)
        # self.sementic_5 = Conv_layer(num*4, num*4, 3, 1, 1)
        # self.sementic_6 = Conv_layer(num*4, num*4, 3, 1, 1)
        # self.trans_3 = CAS(num*4)
        # self.sementic_7 = Conv_layer(num*8, num*8, 3, 1, 1)
        # self.sementic_8 = Conv_layer(num*8, num*8, 3, 1, 1)
        # self.trans_4 = CAS(num*8)
        # self.sementic_9 = Conv_layer(num*16, num*16, 3, 1, 1)
        # self.sementic_10 = Conv_layer(num*16, num*16, 3, 1, 1)
        # self.trans_5 = CAS(num*16)

        # decoder
        # self.up_1 = CAS(num*32, True)
        self.up = nn.PixelShuffle(2)
        self.attention1 = DeBlock(num*8)
        self.attention2 = DeBlock(num*4)
        self.attention3 = DeBlock(num*2)
        self.attention4 = DeBlock(num*1)
        self.attention5 = DeBlock(16)
        self.decoder_10 = nn.Conv2d(16, dim_out, 3, 1, 1)
        # self.up_5 = CAS(num*3, True)
        # self.decoder_9 = Conv_layer(num*3, num*1, 3, 1, 1)
        # self.decoder_10 = nn.Conv2d(num, dim_out, 3, 1, 1)

    def encoder(self, x, y):
        x = torch.cat([x, y], 1)         #[1,6,512,512]
        x_1 = self.encoder_1(x)          #[1,32,512,512]
        x_2 = self.encoder_2(x_1)          #[1,32,512,512]
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
        # x_encoder = x_10                  #[1,512,16,16]
        return x_2, x_4, x_6, x_8, x_10

    def sementic(self, coarse, sementic, x_2, x_4, x_6,x_8):  #coarse:[1,3,512,512] sementic:[1,1,512,512]
        # x1 = torch.cat([coarse, sementic], 1)    #[1,4,512,512]
        # # w = F.softmax(self.conv(x1), dim=1) #[1,4,512,512]
        # # result = coarse*w[:, 0:3, :, :] + sementic*w[:, 3:4, :, :]   #[1,3,512,512]
        # s_2 = self.sementic_2(x1)              #[1,32,512,512]
        # s_2 = self.trans_1(s_2)                  #[1,64,256,256]
        s_2 = self.ffm(coarse, sementic)      #[1,32,256,256]
        x2 = torch.cat([s_2, x_2], 1)         #[1,64,256,256]
        # s_3 = self.sementic_3(x2)                          #[1,64,256,256]
        s_4 = self.sementic_4(x2)             #[1,64,128,128]
        # s_4 = self.trans_2(s_4)                  #[1,128,128,128]
        x3 = torch.cat([s_4, x_4], 1)         #[1,128,128,128]
        # s_5 = self.sementic_5(x3)                          #[1,128,128,128]
        s_6 = self.sementic_6(x3)             #[1,128,64,64]
        # s_6 = self.trans_3(s_6)                  #[1,256,64,64]
        x4 = torch.cat([s_6, x_6], 1)         #[1,256,64,64]
        # s_7 = self.sementic_7(x4)                          #[1,256,64,64] 
        s_8 = self.sementic_8(x4)             #[1,256,64,64]
        # s_8 = self.trans_4(s_8)                  #[1,512,32,32]
        x5 = torch.cat([s_8, x_8], 1)         #[1,512,32,32]
        # s_9 = self.sementic_9(x5)                          #[1,512,32,32]
        s_10 = self.sementic_10(x5)           #[1,512,32,32]
        # s_10 = self.trans_5(s_10)                          #[1,512,16,16]
        # s_encoder = s_10
        return s_2, s_4, s_6, s_8, s_10
    # def sementic(self, coarse, sementic, x_2, x_4, x_6, x_8):  #coarse:[1,3,512,512] sementic:[1,1,512,512]
    #     x1 = torch.cat([coarse, sementic], 1)              #[1,4,512,512]
    #     s_1 = self.sementic_1(x1)                          #[1,32,512,512]
    #     s_2 = self.sementic_2(s_1)                         #[1,32,512,512]
    #     s_2 = self.trans_1(s_2)                            #[1,32,256,256]
    #     x2 = torch.cat([s_2, x_2], 1)                      #[1,64,256,256]
    #     s_3 = self.sementic_3(x2)                          #[1,64,256,256]
    #     s_4 = self.sementic_4(s_3)                         #[1,64,256,256]
    #     s_4 = self.trans_2(s_4)                            #[1,64,128,128]
    #     x3 = torch.cat([s_4, x_4], 1)                      #[1,128,128,128]
    #     s_5 = self.sementic_5(x3)                          #[1,128,128,128]
    #     s_6 = self.sementic_6(s_5)                         #[1,128,128,128]
    #     s_6 = self.trans_3(s_6)                            #[1,128,64,64]
    #     x4 = torch.cat([s_6, x_6], 1)                      #[1,256,64,64]
    #     s_7 = self.sementic_7(x4)                          #[1,256,64,64] 
    #     s_8 = self.sementic_8(s_7)                         #[1,256,64,64]
    #     s_8 = self.trans_4(s_8)                            #[1,256,32,32]
    #     x5 = torch.cat([s_8, x_8], 1)                      #[1,512,32,32]
    #     s_9 = self.sementic_9(x5)                          #[1,512,32,32]
    #     s_10 = self.sementic_10(s_9)                       #[1,512,32,32]
    #     s_10 = self.trans_5(s_10)                          #[1,512,16,16]
    #     s_encoder = s_10
    #     return s_2, s_4, s_6, s_8, s_10
        

    def decoder(self, x, s_2, s_4, s_6, s_8, s_10):
        x_11 = self.up(torch.cat([x, s_10], 1))          #[1,256,32,32]      
        x_12 = self.attention1(x_11)                     #[1,256,32,32]
        # x_12 = self.decoder_2(x_12)                        #[1,512,32,32]
        x_13 = self.up(torch.cat([x_12, s_8], 1))        #[1,128,64,64]
        x_14 = self.attention2(x_13)                     #[1,128,64,64]
        # x_14 = self.decoder_4(x_14)                        #[1,256,64,64]
        x_15 = self.up(torch.cat([x_14, s_6], 1))        #[1,64,128,128]
        x_16 = self.attention3(x_15)                     #[1,64,128,128]
        # x_16 = self.decoder_6(x_16)                        #[1,192,128,128]
        x_17 = self.up(torch.cat([x_16, s_4], 1))        #[1,32,256,256]
        x_18 = self.attention4(x_17)                     #[1,32,256,256]
        
        # x_18 = self.decoder_8(x_18)                        #[1,64,256,256]
        x_19 = self.up(torch.cat([x_18, s_2], 1))        #[1,16,512,512]
        x_20 = self.attention5(x_19)                     #[1,16,512,512]
        x_20 = self.decoder_10(x_20)                       #[1,3,512,512]
        x_20 = (torch.tanh(x_20) + 1) / 2                #[1,3,512,512]
        return x_20
        
    def forward(self, rain, coarse, sementic):
        rain = F.interpolate(rain, size=(512,512), mode='bilinear', align_corners=True)
        x_2, x_4, x_6, x_8, x_10 = self.encoder(rain, coarse)
        s_2, s_4, s_6, s_8, s_10 = self.sementic(coarse, sementic, x_2, x_4, x_6,x_8)
        x_middle =  self.middle(x_10)                  #[1,512,32,32]
        out = self.decoder(x_middle, s_2, s_4, s_6, s_8, s_10) #[1,3,512,512]
        out = F.interpolate(out, size=(500,500), mode='bilinear', align_corners=True)
        return out

# class u_net_heavy(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super(u_net_heavy, self).__init__()       
#         # initial
#         num = 32       
#         # encoder
#         self.encoder_1 = Conv_layer(dim_in,num*1, 7, 1, 3)
#         self.encoder_2 = ERB(num*1, 2)
#         self.down_1 = CAS(num*1)
#         self.encoder_3 = Conv_layer(num*1, num*2, 3, 1, 1)
#         self.encoder_4 = ERB(num*2, 2)
#         self.down_2 = CAS(num*2)
#         self.encoder_5 = Conv_layer(num*2, num*4, 3, 1, 1)
#         self.encoder_6 = ERB(num*4, 2)
#         self.down_3 = CAS(num*4)
#         self.encoder_7 = Conv_layer(num*4, num*8, 3, 1, 1)
#         self.encoder_8 = ERB(num*8, 2)
#         self.down_4 = CAS(num*8)
#         self.encoder_9 = Conv_layer(num*8, num*16, 3, 1, 1)
#         self.encoder_10 = ERB(num*16, 2)
#         self.down_5 = CAS(num*16)

#         # Middle
#         # blocks = []
#         # for _ in range(4):
#         #     block = res_bolock(num*16)
#         #     blocks.append(block)
#         # self.middle = nn.Sequential(*blocks)
#         self.middle = AtrousBlockPad2(in_channels= num*16, out_channels= num*16, kernel_size=3, stride=1, use_bias=True, use_scale=True, activation=nn.ReLU(True))


#         # sementic
#         self.sementic_1 = Conv_layer(1+3,num*1, 7, 1, 3)
#         self.sementic_2 = Conv_layer(num*1, num*1, 3, 1, 1)
#         self.trans_1 = CAS(num*1)
#         self.sementic_3 = Conv_layer(num*2, num*2, 3, 1, 1)
#         self.sementic_4 = Conv_layer(num*2, num*2, 3, 1, 1)
#         self.trans_2 = CAS(num*2)
#         self.sementic_5 = Conv_layer(num*4, num*4, 3, 1, 1)
#         self.sementic_6 = Conv_layer(num*4, num*4, 3, 1, 1)
#         self.trans_3 = CAS(num*4)
#         self.sementic_7 = Conv_layer(num*8, num*8, 3, 1, 1)
#         self.sementic_8 = Conv_layer(num*8, num*8, 3, 1, 1)
#         self.trans_4 = CAS(num*8)
#         self.sementic_9 = Conv_layer(num*16, num*16, 3, 1, 1)
#         self.sementic_10 = Conv_layer(num*16, num*16, 3, 1, 1)
#         self.trans_5 = CAS(num*16)

#         # decoder
#         self.up_1 = CAS(num*32, True)
#         self.decoder_1 = Conv_layer(num*32, num*32, 3, 1, 1)
#         self.decoder_2 = Conv_layer(num*32, num*16, 3, 1, 1)
#         self.up_2 = CAS(num*24, True)
#         self.decoder_3 = Conv_layer(num*24, num*24, 3, 1, 1)
#         self.decoder_4 = Conv_layer(num*24, num*8, 3, 1, 1)
#         self.up_3 = CAS(num*12, True)
#         self.decoder_5 = Conv_layer(num*12, num*12, 3, 1, 1)
#         self.decoder_6 = Conv_layer(num*12, num*6, 3, 1, 1)
#         self.up_4 = CAS(num*8, True)
#         self.decoder_7 = Conv_layer(num*8, num*8, 3, 1, 1)
#         self.decoder_8 = Conv_layer(num*8, num*2, 3, 1, 1)
#         self.up_5 = CAS(num*3, True)
#         self.decoder_9 = Conv_layer(num*3, num*1, 3, 1, 1)
#         self.decoder_10 = nn.Conv2d(num, dim_out, 3, 1, 1)

#     def encoder(self, x, y):
#         x = torch.cat([x, y], 1)         #[1,6,512,512]
#         x_1 = self.encoder_1(x)          #[1,32,512,512]
#         x_2 = self.encoder_2(x_1)        #[1,32,512,512]
#         x_2 = self.down_1(x_2)           #[1,32,256,256]
#         x_3 = self.encoder_3(x_2)        #[1,64,256,256]
#         x_4 = self.encoder_4(x_3)        #[1,64,256,256]
#         x_4 = self.down_2(x_4)           #[1,64,128,128]
#         x_5 = self.encoder_5(x_4)         #[1,128,128,128]
#         x_6 = self.encoder_6(x_5)         #[1,128,128,128]
#         x_6 = self.down_3(x_6)            #[1,128,64,64]
#         x_7 = self.encoder_7(x_6)         #[1,256,64,64]
#         x_8 = self.encoder_8(x_7)         #[1,256,64,64]
#         x_8 = self.down_4(x_8)            #[1,256,32,32]
#         x_9 = self.encoder_9(x_8)         #[1,512,32,32]
#         x_10 = self.encoder_10(x_9)       #[1,512,32,32]
#         x_10 = self.down_5(x_10)          #[1,512,16,16]
#         x_encoder = x_10                  #[1,512,16,16]
#         return x_2, x_4, x_6, x_8, x_10

#     def sementic(self, coarse, sementic, x_2, x_4, x_6, x_8):  #coarse:[1,3,512,512] sementic:[1,1,512,512]
#         x1 = torch.cat([coarse, sementic], 1)              #[1,4,512,512]
#         s_1 = self.sementic_1(x1)                          #[1,32,512,512]
#         s_2 = self.sementic_2(s_1)                         #[1,32,512,512]
#         s_2 = self.trans_1(s_2)                            #[1,32,256,256]
#         x2 = torch.cat([s_2, x_2], 1)                      #[1,64,256,256]
#         s_3 = self.sementic_3(x2)                          #[1,64,256,256]
#         s_4 = self.sementic_4(s_3)                         #[1,64,256,256]
#         s_4 = self.trans_2(s_4)                            #[1,64,128,128]
#         x3 = torch.cat([s_4, x_4], 1)                      #[1,128,128,128]
#         s_5 = self.sementic_5(x3)                          #[1,128,128,128]
#         s_6 = self.sementic_6(s_5)                         #[1,128,128,128]
#         s_6 = self.trans_3(s_6)                            #[1,128,64,64]
#         x4 = torch.cat([s_6, x_6], 1)                      #[1,256,64,64]
#         s_7 = self.sementic_7(x4)                          #[1,256,64,64] 
#         s_8 = self.sementic_8(s_7)                         #[1,256,64,64]
#         s_8 = self.trans_4(s_8)                            #[1,256,32,32]
#         x5 = torch.cat([s_8, x_8], 1)                      #[1,512,32,32]
#         s_9 = self.sementic_9(x5)                          #[1,512,32,32]
#         s_10 = self.sementic_10(s_9)                       #[1,512,32,32]
#         s_10 = self.trans_5(s_10)                          #[1,512,16,16]
#         s_encoder = s_10
#         return s_2, s_4, s_6, s_8, s_10

#     def decoder(self, x, s_2, s_4, s_6, s_8, s_10):
#         x_11 = self.up_1(torch.cat([x, s_10], 1))          #[1,1024,32,32]      
#         x_12 = self.decoder_1(x_11)                        #[1,1024,32,32]
#         x_12 = self.decoder_2(x_12)                        #[1,512,32,32]
#         x_13 = self.up_2(torch.cat([x_12, s_8], 1))        #[1,768,64,64]
#         x_14 = self.decoder_3(x_13)                        #[1,768,64,64]
#         x_14 = self.decoder_4(x_14)                        #[1,256,64,64]
#         x_15 = self.up_3(torch.cat([x_14, s_6], 1))        #[1,384,128,128]
#         x_16 = self.decoder_5(x_15)                        #[1,384,128,128]
#         x_16 = self.decoder_6(x_16)                        #[1,192,128,128]
#         x_17 = self.up_4(torch.cat([x_16, s_4], 1))        #[1,256,256,256]
#         x_18 = self.decoder_7(x_17)                        #[1,256,256,256]
#         x_18 = self.decoder_8(x_18)                        #[1,64,256,256]
#         x_19 = self.up_5(torch.cat([x_18, s_2], 1))        #[1,96,512,512]
#         x_20 = self.decoder_9(x_19)                        #[1,32,512,512]
#         x_20 = self.decoder_10(x_20)                       #[1,3,512,512]
#         x_20 = (torch.tanh(x_20) + 1) / 2                  #[1,3,512,512]
#         return x_20
        
#     def forward(self, rain, coarse, sementic):
#         x_2, x_4, x_6, x_8, x_10 = self.encoder(rain, coarse)
#         s_2, s_4, s_6, s_8, s_10 = self.sementic(coarse, sementic, x_2, x_4, x_6, x_8)
#         x_middle =  self.middle(x_10)                  #[1,512,16,16]
#         out = self.decoder(x_middle, s_2, s_4, s_6, s_8, s_10) #[1,3,512,512]
#         return out
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x1 = torch.randn(1,3,512,512)
    x2 = torch.randn(1,3,512,512)
    x3 = torch.randn(1,1,512,512)
    unet = u_net_heavy(6, 3)
    out2 = unet(x1,x2,x3)
    print(out2.shape)
# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.randn(1,3,256,256)
#     unet = UNet(3,3)
#     out1,out2 = unet(x)
#     print(out1.shape,out2.shape)