import argparse
import datetime
import itertools
import math
import os
import sys
import time
import lpips
# import colour
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

from datasets import *
import utils
# from utils import ext_transforms as et
from dataset.cityscapes import Cityscapes,Cityscapes_Dark
from dataset import cityscapes
from models.model_UNet import *
from models.ERFNet import *
import models.SegNet as seg
from models.models_x import *
from metrics import *

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0,
                    help="epoch to start training from, 0 starts from scratch, >0 starts from saved checkpoints")
parser.add_argument("--n_epochs", type=int, default=100, help="total number of epochs of training")
# parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
# parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--lambda_smooth", type=float, default=0.0001, help="smooth regularization")
# parser.add_argument("--lambda_monotonicity", type=float, default=10.0, help="monotonicity regularization")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
parser.add_argument("--model_dir", type=str, default="./initmodel",help="path to save result")
parser.add_argument("--output_dir", type=str, default="Model-2",help="path to save model")
parser.add_argument("--result_dir", type=str, default="result",help="path to save result")
parser.add_argument("--image_dir", type=str, default='./save_images/LOL',help="path to Dataset")
parser.add_argument("--data_root", type=str, default='./dataset/LOL',help="path to Dataset")
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--crop', action='store_true')
parser.add_argument('--crop_size', type=int, default=360, help='Takes effect when using --crop ')
# parser.add_argument("--crop_size", type=int, default=384)
opt = parser.parse_args()
crop_size = 'whole_img'
if opt.crop:
    crop_size = opt.crop_size
os.makedirs("saved_models/%s" % opt.output_dir, exist_ok=True)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Loss functions
# criterion_pixelwise = torch.nn.MSELoss()
mse_loss = torch.nn.MSELoss()
seg_loss = torch.nn.L1Loss()
# ssim = utils.loss.SSIM()
ssim_loss = utils.loss.SSIM()
per_loss = utils.loss.perception_loss()
# criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

# Initialize generator and discriminator
LUT0 = Generator3DLUT_identity()
LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()

# net = LHSNet()
seg_net = ERFNet(19)
# seg_net = seg.SegNet(19)
net = u_net_heavy(6,3)
# unet = UNet(3,3)
# classifier = Classifier()
TV3 = TV_3D()
trilinear_ = TrilinearInterpolation()

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    seg_net = seg_net.cuda()
    net = net.cuda()
    # unet = unet.cuda()
    ssim_loss.cuda()
    per_loss.cuda()
    mse_loss.cuda()
    seg_loss.cuda()
    # LUT0 = LUT0.to(device)
    # LUT1 = LUT1.to(device)
    # LUT2 = LUT2.to(device)
    # seg_net = seg_net.to(device)
    # net = net.to(device)
    # ssim_loss.to(device)
    # per_loss.to(device)
    # mse_loss.to(device)
    # seg_loss.to(device)

if opt.epoch != 0:
    # Load pretrained models
    LUTs = torch.load("saved_models/%s/LUTs_6.pth" % (opt.output_dir))
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    # classifier.load_state_dict(torch.load("saved_models/%s/classifier_%d.pth" % (opt.output_dir, opt.epoch)))
    net.load_state_dict(torch.load("saved_models/%s/model_6.pth" % (opt.output_dir)))
    seg_net.load_state_dict(torch.load("saved_models/%s/segNet_6.pth" % (opt.output_dir)))
# else:
#     # Initialize weights
#     # classifier.apply(weights_init_normal_classifier)
#     # torch.nn.init.constant_(classifier.model[16].bias.data, 1.0)
    # net.apply(weights_init_normal_classifier)
    # torch.nn.init.constant_(net, 1.0)

# Optimizers
# ignored_params = list(map(id, seg_net.dec.parameters()))
# params = filter(lambda p: id(p) not in ignored_params, seg_net.parameters())
# optimizer_G = torch.optim.Adam(params, lr=opt.lr,betas=(opt.b1, opt.b2))  # , LUT3.parameters(), LUT4.parameters()
checkpoint = './initmodel/ERFNet_165.pth'
# model = torch.load(checkpoint)
seg_net.load_state_dict(torch.load(checkpoint),strict=False)
for name, param in seg_net.named_parameters():
    if "dec" in name:
        param.requires_grad = False
# for value1 ,value2 in zip(model.items(), seg_net.state_dict().items()):
#     print(value1,value2)
param = filter(lambda p:p.requires_grad, seg_net.parameters())
optimizer = torch.optim.Adam(param, lr=opt.lr,betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(
    # itertools.chain(classifier.parameters(), LUT0.parameters(), LUT1.parameters(), LUT2.parameters()), lr=opt.lr,
    itertools.chain(net.parameters(), LUT0.parameters(), LUT1.parameters(), LUT2.parameters()), lr=opt.lr,
    betas=(opt.b1, opt.b2))  # , LUT3.parameters(), LUT4.parameters()

# train_dst = Cityscapes(root=opt.data_root + '/train',
#                         train=True, size=opt.crop_size)
# test_dst = Cityscapes(root=opt.data_root + '/test',
#                         train=False, size= 'whole_img')
train_dst = Cityscapes_Dark(root='./dataset/LOL/train',
                        train=True , size=opt.crop_size)
test_dst = Cityscapes_Dark(root='./dataset/LOL/test',
                        train=False , size= 'whole_img')
train_loader = DataLoader(train_dst, batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(test_dst, batch_size=opt.batch_size, shuffle=False)


def generator_train(img):
    # pred,out = unet(img)
    pred,out = seg_net(img)  #[1,128,64,64]->[1,3,1,1],[1,19,512,512]
    # save_image(out, './out/3.jpg')
    pred = pred.squeeze()
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT
    # weights_norm = torch.mean(pred ** 2)
    combine_A = img.new(img.size())
    _, combine_A = trilinear_(LUT, F.interpolate(img, size=(512,512), mode='bilinear', align_corners=True))
    return combine_A,out


def generator_eval(img):
    # pred,out = unet(img)
    # pred, out = net(img).squeeze()
    pred, out = seg_net(img)  # [1,128,64,64]->[1,3,1,1],[1,3,256,256]
    # save_image(out, "./out/1.png")
    pred = pred.squeeze()
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT
    # weights_norm = torch.mean(pred ** 2)
    combine_A = img.new(img.size())
    # _,combine_A = trilinear_(LUT, img)
    _, combine_A = trilinear_(LUT, F.interpolate(img, size=(512,512), mode='bilinear', align_corners=True))
    return combine_A, out

# ----------
#  Training
# ----------
max_ssim = 0
max_psnr = 0
ssims = []
psnrs = []
max_epoch = 0
losses = []
for epoch in range(opt.epoch, opt.n_epochs):
    mse_avg = 0
    psnr_avg = 0
    epoch_start_time = time.time()
    # classifier.train()
    seg_net.train()
    net.train()
    for i, (images,targets,image_name) in enumerate(train_loader):
        # Model inputs
        images = images.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)
        # save_image(torch.cat([images, targets], dim=0), './predict/2.jpg')

        # net.zero_grad()
        optimizer.zero_grad()
        optimizer_G.zero_grad()

        # coarse_pred,seg_low = generator_train(images)
        # _,seg_pred = net(coarse_pred,None, None, True)
        t0 = time.time()
        coarse_pred,seg_low = generator_train(images) #[1,3,512,512] [1,19,512,512]
        _,seg_pred = seg_net(coarse_pred)             #[1,19,512,512]
        _,seg_high = seg_net(targets)                 #[1,19,512,512]
        # _,seg_pred = net(coarse_pred,None, None, True)
        # _,seg_high = net(targets,None, None, True)
        seg_gray = seg_low.argmax(dim=1)              #[1,512,512]   #torch.argmax返回指定维度最大值的序号
        seg_gray = seg_gray.unsqueeze(1).float()      #[1,1,512,512]
        # save_image(torch.cat([images, coarse_pred, targets], dim=0), './predict/2.jpg')
        # refined_pred = net(images, coarse_pred, seg_gray, False)
        refined_pred = net(images, coarse_pred, seg_gray)
        save_image(torch.cat([images, refined_pred, targets], dim=0), './predict/train.jpg')
        # Pixel-wise loss
        
        mse = mse_loss(coarse_pred, targets)
        # loss_seg = seg_loss(seg_pred,seg_high) / (seg_loss(seg_pred,seg_low) + 1e-4)
        # seg = criterion(seg_pred, seg_high) 
        loss_ssim = ssim_loss(refined_pred, targets)
        # loss_ssim = ssim_loss(coarse_pred, targets)
        loss_per = per_loss(refined_pred, targets)
        
        loss = mse + loss_ssim + loss_per
        # loss = mse + 0.1*loss_seg + loss_ssim + loss_per
        # loss_seg.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        optimizer_G.step()
        t1 = time.time()
        losses.append(loss.item())

        # Print log
        sys.stdout.write(
            "\r[Epoch: %d/%d] [Batch: %d/%d] [time: %f]"
            % (epoch, opt.n_epochs, i, len(train_loader), t1-t0))

    if epoch % (opt.snapshots) == 0:
        seg_net.eval()
        net.eval()
        # generator_eval.eval()
        ssims = []
        psnrs = []
        lpips_score = 0
        loss_fn_alex = lpips.LPIPS(net='alex').cuda()
            # save_image(torch.cat([inputs, targets], dim=0), './out/3.jpg')
        with torch.no_grad():
            for i, (inputs, targets, image_name) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                img_id = 0
                image_name = image_name[0].split('/')[-1].split('.')[0]
                coarse_pred,seg_low = generator_eval(inputs)
                _,seg_pred = seg_net(coarse_pred)                 
                seg_gray = seg_low.argmax(dim=1) #[1,512,512]   #torch.argmax返回指定维度最大值的序号
                
                # image = inputs.squeeze(0)        
                # image = image.detach().cpu().numpy()                
                # image = (image * 255).transpose(1, 2, 0).astype(np.uint8)
                # # pred = test_loader.dataset.decode_target(pred).astype(np.uint8)
                # prediction = seg_pred.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()#(512,512) 
                # # prediction = cityscapes.colorize_mask(prediction)#(512,512)
                # prediction = Cityscapes.decode_target(prediction).astype(np.uint8)
                # Image.fromarray(prediction).save('save_images/semantic/%d_prediction.png' % img_id)               
                # Image.fromarray(image).save('save_images/semantic/%d_image.png' % img_id)
                # Image.fromarray(pred).save('save_images/semantic/%d_target.png' % img_id)

                seg_gray = seg_gray.unsqueeze(1).float() 
                refined_pred = net(inputs, coarse_pred, seg_gray)
                save_image(refined_pred, os.path.join(opt.image_dir, "%s.png" % (image_name)), nrow=1, normalize=False)
                # save_image(torch.cat([refined_pred, targets], dim=0), './predict/test.jpg')
                # 精度计算
                # ssim1 = ssim(coarse_pred, targets).item()
                # psnr1 = psnr(coarse_pred, targets)
                ssim1 = ssim(refined_pred, targets).item()
                psnr1 = psnr(refined_pred, targets)
                lpips1 = loss_fn_alex(refined_pred, targets)

                # def ave(lis):
                #     s = 0
                #     total_num = len(lis)
                #     for i in lis:
                #         s = s + i
                #     return s/total_num

                ssims.append(ssim1)
                psnrs.append(psnr1)
                lpips_score = lpips_score + lpips1
            ssim_eval = np.mean(ssims)
            psnr_eval = np.mean(psnrs)
            lpips_eval = float(lpips_score) / i
            print(f'\npsnr:{psnr_eval:.4f}\t|ssim:{ssim_eval:.4f}\t|lpips:{lpips_eval:.4f}')

        # ssims.append(ssim_eval)
        # psnrs.append(psnr_eval)      
        if ssim_eval > max_ssim or psnr_eval > max_psnr:
            max_ssim = max(max_ssim, ssim_eval)
            max_psnr = max(max_psnr, psnr_eval)
            max_epoch = epoch

            # Save model checkpoints
            LUTs = {"0": LUT0.state_dict(), "1": LUT1.state_dict(),
                    "2": LUT2.state_dict()}  # ,"3": LUT3.state_dict(),"4": LUT4.state_dict()
            torch.save(LUTs, "saved_models/%s/LUTs_%d.pth" % (opt.output_dir, epoch))
            torch.save(seg_net.state_dict(), "saved_models/%s/segNet_%d.pth" % (opt.output_dir, epoch))
            torch.save(net.state_dict(), "saved_models/%s/model_%d.pth" % (opt.output_dir, epoch))
            # save_image(refined_pred, os.path.join(opt.image_dir, "%s.png" % (image_name)), nrow=1, normalize=False)
            # file = open('saved_models/%s/result.txt' % opt.result_dir, 'a')
            # file.write(" [max PSNR: %f] [max_SSIM: %f, max_epoch: %d]\n" % (max_psnr, max_ssim, max_epoch))
            # file.close()

            print("max_epoch:{}\t|max psnr:{:.4f}\t|max_ssim:{:.4f}\t|Time:{:.4f}\t|Loss: {:.4f}\t".format(max_epoch,max_psnr, max_ssim, time.time() - epoch_start_time, loss.item()))#\t代表四个空格
            print("----------------------------------------------------------")
    # if (epoch+1) % 10 == 0:
    #    visualize_result(epoch+1)


