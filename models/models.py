import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.ESPNet import ESPNet
# from models.shelfnet import ShelfNet
# from models.SegNet import SegNet

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
            nn.Conv2d(in_dim, out_dim, 3, 1, 1), 
            nn.LeakyReLU(inplace=True),
            Res2NetBottleneck(out_dim, out_dim),
        )
        self.down = nn.Conv2d(out_dim, out_dim, 3, 2, 1)

    def forward(self,x): 
        x = self.conv(x) 
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
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes, bottleneck_planes // scales, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes) for _ in range(scales-1)])
        self.conv3 =nn.Conv2d(bottleneck_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.LeakyReLU(inplace=True)
        self.sa = SA() if sa else None
        self.ca = CA(planes) if ca else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x       

        out = self.conv1(x)  
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1) 
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])       
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s])))) 
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1])))) 
        if self.sa is not None:
            ys_0 = self.sa(ys[0])
            ys_1 = self.sa(ys[1])
            ys_2 = self.sa(ys[2])
            ys_3 = self.sa(ys[3])
 
        out = torch.cat([ys_0,ys_1,ys_2,ys_3], 1)
        if self.ca is not None:
            out = self.ca(out) 

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity        
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

    def forward(self, input):
        net = self.conv(input) 
        net = self.conv_aftermath(net)
        return net


class CAS(nn.Module):
    # context-aware scaling
    def __init__(self, channels,is_up=False):
        super(CAS, self).__init__()
        self.context = CA(channels)      
        self.down = nn.Conv2d(channels, channels, 3, 1, 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.is_up = is_up
        
    def forward(self, x):
        con_attention = self.context(x) 
        if self.is_up == True:
            x = self.up(x)
        else:
            x = self.down(con_attention) 
        return x   

class DAB(nn.Module):
    def __init__(self, channels, reduction=16):
        super(DAB, self).__init__()
      
        self.SA = spatial_attn_layer()       
        self.CA = CA(channels, reduction)     
        self.conv1x1 = nn.Conv2d(channels+1, channels, kernel_size=1)


    def forward(self, x): 
        res = x               
        sa_branch = self.SA(res) 
        ca_branch = self.CA(res) 
        res = torch.cat([sa_branch, ca_branch], dim=1) 
        res = self.conv1x1(res)  
        res += x               
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

    def forward(self, x):   
        att = self.body(x)  
        out = att * x       
        return out

class FFModule(nn.Module):
    def __init__(self,out_channels):
        super(FFModule, self).__init__()

        self.conv = conv3x3(3, 4)
        self.fuse1 = nn.Conv2d(3, out_channels, 1, 1, 0)
        self.fuse2 = nn.Conv2d(1, out_channels, 1, 1, 0)
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0)
        self.conv2 = CA(out_channels)      
        self.down = nn.Conv2d(out_channels, out_channels, 3, 2, 1)

    def forward(self, x1,x2):

        x = torch.cat([x1,x2], 1)
        w = F.softmax(self.conv(x), dim=1) 
        x1 = x1 * w[:, 0:3, :, :]
        x2 = x2 * w[:, 3:4, :, :] 
        x = torch.cat([self.fuse1(x1),self.fuse2(x2)], 1) 

        x = self.fuse(x) 
        x = self.conv2(x) 
        y = self.down(x) 
        return y

class DeBlock(nn.Module):
    def __init__(self, dim, kernel_size=3,stride=1,padding=1):
        super(DeBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size,stride, padding, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size,stride, padding, bias=True)
        self.calayer = CA(dim)
        self.palayer = PA(dim)

    def forward(self, x):             
        res = self.act1(self.conv1(x))  
        res1 = res + x                   
        res2 = self.act1(self.conv2(res1)) 
        res3 = res + res2 
        res4 = self.calayer(res3)        
        res5 = self.palayer(res4)        
        res = res5 + res2                     

        return res

class u_net_heavy(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(u_net_heavy, self).__init__()       
        # initial
        num = 32       

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

  
        self.sementic_2 = Res2NetConv(1+3,num*1)
  
        self.ffm = FFModule(num*1)

        self.sementic_4 = Res2NetConv(num*2, num*2)

        self.sementic_6 = Res2NetConv(num*4, num*4)
  
        self.sementic_8 = Res2NetConv(num*8, num*8)

        self.sementic_10 = Res2NetConv(num*16, num*16)
        self.up = nn.PixelShuffle(2)
        self.attention1 = DeBlock(num*8)
        self.attention2 = DeBlock(num*4)
        self.attention3 = DeBlock(num*2)
        self.attention4 = DeBlock(num*1)
        self.attention5 = DeBlock(16)
        self.decoder_10 = nn.Conv2d(16, dim_out, 3, 1, 1)


    def encoder(self, x, y):
        x = torch.cat([x, y], 1)         
        x_1 = self.encoder_1(x)          
        x_2 = self.encoder_2(x_1)          
        x_2 = self.down_1(x_2)           
        x_3 = self.encoder_3(x_2)        
        x_4 = self.encoder_4(x_3)        
        x_4 = self.down_2(x_4)           
        x_5 = self.encoder_5(x_4)         
        x_6 = self.encoder_6(x_5)         
        x_6 = self.down_3(x_6)            
        x_7 = self.encoder_7(x_6)         
        x_8 = self.encoder_8(x_7)        
        x_8 = self.down_4(x_8)         
        x_9 = self.encoder_9(x_8)        
        x_10 = self.encoder_10(x_9)      
        x_10 = self.down_5(x_10)         
        return x_2, x_4, x_6, x_8, x_10

    def sementic(self, coarse, sementic, x_2, x_4, x_6,x_8):
        s_2 = self.ffm(coarse, sementic)      
        x2 = torch.cat([s_2, x_2], 1)         
        s_4 = self.sementic_4(x2)                    
        x3 = torch.cat([s_4, x_4], 1)       
        s_6 = self.sementic_6(x3)             
        x4 = torch.cat([s_6, x_6], 1)       
        s_8 = self.sementic_8(x4)                     
        x5 = torch.cat([s_8, x_8], 1)                         
        s_10 = self.sementic_10(x5)          
        return s_2, s_4, s_6, s_8, s_10

        
    def decoder(self, x, s_2, s_4, s_6, s_8, s_10):
        x_11 = self.up(torch.cat([x, s_10], 1))             
        x_12 = self.attention1(x_11)                                           
        x_13 = self.up(torch.cat([x_12, s_8], 1))        
        x_14 = self.attention2(x_13)                                            
        x_15 = self.up(torch.cat([x_14, s_6], 1))       
        x_16 = self.attention3(x_15)                                            
        x_17 = self.up(torch.cat([x_16, s_4], 1))        
        x_18 = self.attention4(x_17)                    
        x_19 = self.up(torch.cat([x_18, s_2], 1))        
        x_20 = self.attention5(x_19)                    
        x_20 = self.decoder_10(x_20)                     
        x_20 = (torch.tanh(x_20) + 1) / 2                
        return x_20
        
    def forward(self, rain, coarse, sementic):
        b,c,h,w = rain.size()
        rain = F.interpolate(rain, size=(512,512), mode='bilinear', align_corners=True)
        x_2, x_4, x_6, x_8, x_10 = self.encoder(rain, coarse)
        s_2, s_4, s_6, s_8, s_10 = self.sementic(coarse, sementic, x_2, x_4, x_6,x_8)
        x_middle =  self.middle(x_10)                  
        out = self.decoder(x_middle, s_2, s_4, s_6, s_8, s_10) 
        return out