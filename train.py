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
parser.add_argument("--output_dir", type=str, default="",help="path to save model")
parser.add_argument("--result_dir", type=str, default="result",help="path to save result")
parser.add_argument("--image_dir", type=str, default='./save_images/',help="path to Dataset")
parser.add_argument("--data_root", type=str, default='./dataset/',help="path to Dataset")
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--crop', action='store_true')
parser.add_argument('--crop_size', type=int, default=512, help='Takes effect when using --crop ')
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
mse_loss = torch.nn.MSELoss()
seg_loss = torch.nn.L1Loss()
ssim_loss = utils.loss.SSIM()
per_loss = utils.loss.perception_loss()

# Initialize generator and discriminator
LUT0 = Generator3DLUT_identity()
LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()

seg_net = ERFNet()
net = u_net_heavy(6,3)
TV3 = TV_3D()
trilinear_ = TrilinearInterpolation()

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    seg_net = seg_net.cuda()
    net = net.cuda()
    ssim_loss.cuda()
    per_loss.cuda()
    mse_loss.cuda()
    seg_loss.cuda()


checkpoint = './initmodel/erf.pth'
seg_net.load_state_dict(torch.load(checkpoint),strict=False)
for name, param in seg_net.named_parameters():
    if "dec" in name:
        param.requires_grad = False

param = filter(lambda p:p.requires_grad, seg_net.parameters())
optimizer = torch.optim.Adam(param, lr=opt.lr,betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(
    itertools.chain(net.parameters(), LUT0.parameters(), LUT1.parameters(), LUT2.parameters()), lr=opt.lr,
    betas=(opt.b1, opt.b2))

train_dst = Cityscapes(root=opt.data_root + '/train',
                        train=True, size=opt.crop_size)
test_dst = Cityscapes(root=opt.data_root + '/val',
                        train=False, size= 'whole_img')

train_loader = DataLoader(train_dst, batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(test_dst, batch_size=opt.batch_size, shuffle=False)


def generator_train(img):
    pred,out = seg_net(img)  #[1,128,64,64]->[1,3,1,1],[1,19,512,512]
    pred = pred.squeeze()
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT
    combine_A = img.new(img.size())
    _, combine_A = trilinear_(LUT, img)
    return combine_A,out


def generator_eval(img):
    pred, out = seg_net(img) 
    pred = pred.squeeze()
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT
    combine_A = img.new(img.size())
    _, combine_A = trilinear_(LUT, img)
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
    seg_net.train()
    net.train()
    for i, (images,targets,image_name) in enumerate(train_loader):
        # Model inputs
        images = images.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        optimizer_G.zero_grad()

        t0 = time.time()
        coarse_pred,seg_low = generator_train(images) 
        _,seg_pred = seg_net(coarse_pred)             
        _,seg_high = seg_net(targets)                 

        # seg_gray = seg_pred.argmax(dim=1)             
        # seg_gray = seg_gray.unsqueeze(1).float()      

        refined_pred = net(images, coarse_pred, seg_gray)

        
        mse = mse_loss(coarse_pred, targets)
        loss_ssim = ssim_loss(refined_pred, targets)
        loss_per = per_loss(refined_pred, targets)
        
        # loss = mse 
        loss = 0.1*mse + loss_ssim + loss_per

        loss.backward()
        optimizer.step()
        optimizer_G.step()
        t1 = time.time()
        losses.append(loss.item())

    if epoch % (opt.snapshots) == 0:
        seg_net.eval()
        net.eval()
    
        with torch.no_grad():
            for i, (inputs, targets, image_name) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                image_name = image_name[0].split('/')[-1].split('.')[0]
                coarse_pred,seg_low = generator_eval(inputs)
                _,seg_pred = seg_net(coarse_pred)                 
                seg_gray = seg_low.argmax(dim=1) 

                seg_gray = seg_gray.unsqueeze(1).float() 
                refined_pred = net(inputs, coarse_pred, seg_gray)

            print("----------------------------------------------------------")

