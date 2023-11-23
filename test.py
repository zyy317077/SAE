import argparse
import datetime
import itertools
import math
import os
import sys
import time
import lpips
import argparse
import time
import torch
import torchvision.transforms as tfs
from niqe import niqe
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.model_UNet import *
from models.ERFNet import *
from models.RUAS import *
from models.models_x import *
from datasets import *
from metrics import *
import lpips
from collections import namedtuple
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=85, help="epoch to load the saved checkpoint")
parser.add_argument("--model_dir", type=str, default=".", help="directory of saved models")
parser.add_argument("--coarse_dir", type=str, default='./save_images/',help="path to Dataset")
parser.add_argument("--fine_dir", type=str, default='./save_images/',help="path to Dataset")
opt = parser.parse_args()
# opt.model_dir = opt.model_dir + '_' + opt.input_color_space

# use gpu when detect cuda
cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

criterion_pixelwise = torch.nn.MSELoss()
LUT0 = Generator3DLUT_identity()
LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()
seg_net = ERFNet()
net = u_net_heavy(6,3)
trilinear_ = TrilinearInterpolation() 

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    seg_net = seg_net.cuda()
    net = net.cuda()

def generator(img):
    pred,out = net(img, None, None, True)
    pred = pred.squeeze()
        
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT 

    combine_A = img.new(img.size())
    _, combine_A = trilinear_(LUT, img)
    return combine_A,out

def test():  
    net.load_state_dict(torch.load("saved_models/%s/model.pth" % (opt.model_dir)), strict=False)
    net.eval()  
    LUTs = torch.load("saved_models/%s/LUTs.pth" % (opt.model_dir))
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    LUT0.eval()
    LUT1.eval()
    LUT2.eval()
    
    with torch.no_grad():
    # """Saves a generated sample from the validation set"""
        test_path = './dataset/LOL/test/low'
        target_path = './dataset/LOL/test/high'
        test_imgs_dir = sorted(os.listdir(test_path))
        target_imgs_dir = sorted(os.listdir(target_path))
        niqes = []
        ssims=[]
        psnrs=[]
        # lpips_eval = []
        lpips_score = 0.0
        for i in range(len(test_imgs_dir)):
            test_name = test_imgs_dir[i]                            #'0016E5_08220_L.png'
            target_name = target_imgs_dir[i]
            test_imgs = os.path.join(test_path, test_name)
            target_imgs = os.path.join(target_path, target_name)
            test_img = Image.open(test_imgs).convert('RGB')
            target_img = Image.open(target_imgs) 
            # test_img = tfs.Resize([512, 512])(test_img)
            # target_img = tfs.Resize([512, 512])(target_img)
            test_img = tfs.ToTensor()(test_img)
            target_img = tfs.ToTensor()(target_img)
            test_img = test_img.reshape(1,test_img.shape[0],test_img.shape[1],test_img.shape[2])
            target_img = target_img.reshape(1,target_img.shape[0],target_img.shape[1],target_img.shape[2])
            test_img = test_img.cuda()    
            target_img = target_img.cuda() 

            # ----------
            #  model
            # ----------
            t_list = [] 
            t0 = time.time()
            pred,seg_pred = seg_net(test_img)
            pred = pred.squeeze()
            LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT 
            _, coarse_pred = trilinear_(LUT, test_img)
            seg_gray = seg_pred.argmax(dim=1)
            seg_gray = seg_gray.unsqueeze(1).float() 
            refined_pred = net(test_img, coarse_pred, seg_gray)

            t1 = time.time()
            t_list.append(t1 - t0)
        
            # ssim1 = ssim(test_img_tensor, target_img_tensor).item()
        #     ssim1 = ssim(refined_pred, target_img).item()
        #     psnr1 = psnr(refined_pred, target_img)
        #     ssims.append(ssim1)
        #     psnrs.append(psnr1)
        # ssim_eval = np.mean(ssims)
        # psnr_eval = np.mean(psnrs)


# ----------
#  evaluation
# ----------
test()

