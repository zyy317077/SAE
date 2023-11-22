import torch
import torch.utils
from PIL import Image
from math import exp
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import os
import torchvision.transforms as tfs
from niqe import niqe
def gaussian(window_size, sigma):

    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):

    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)

def psnr(pred, gt):

    pred = pred.clamp(0, 1).cpu().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)


def main():
    # test_path = '/home2/lhx/SCL-LLE-main/result'
    test_path = '/home2/zyy/Image-Adaptive-3DLUT/save_images/DICM'
    # target_path = '/home2/lhx/Zero-DCE++/data/target'
    test_imgs_dir = os.listdir(test_path)
    # target_imgs_dir = os.listdir(target_path)
    ssims = []
    psnrs = []
    niqes = []
    for i in range(len(test_imgs_dir)):
        test_name = test_imgs_dir[i]
        # target_name = target_imgs_dir[i]
        test_imgs = os.path.join(test_path, test_name)
        # target_imgs = os.path.join(target_path, target_name)
        test_img = Image.open(test_imgs) 
        # target_img = Image.open(target_imgs) 
        test_img = tfs.Resize([360, 360])(test_img)
        test_img_tensor = tfs.ToTensor()(test_img)
        target_img = tfs.Resize([360, 360])(target_img)
        target_img_tensor = tfs.ToTensor()(target_img)
        # test_img_tensor = test_img_tensor.reshape(1,test_img_tensor.shape[0],test_img_tensor.shape[1],test_img_tensor.shape[2])
        # target_img_tensor = target_img_tensor.reshape(1,target_img_tensor.shape[0],target_img_tensor.shape[1],target_img_tensor.shape[2])
        t = np.array(Image.open(test_imgs).convert('LA'))[:,:,0] 
        niqescore = niqe(t)
        # ssim1 = ssim(test_img_tensor, target_img_tensor).item()
        # psnr1 = psnr(test_img_tensor, target_img_tensor)
        # ssims.append(ssim1)
        # psnrs.append(psnr1)
        niqes.append(niqescore)
    # ssim_mean = np.mean(ssims)
    # psnr_mean = np.mean(psnrs)
    niqes_mean = np.mean(niqes)
    print(f'\r NIQE:{niqes_mean:.5f}')
    # print(f'\rSSIM:{ssim_mean:.5f} PSNR:{psnr_mean:.5f} NIQE:{niqes_mean:.5f}')
    

if __name__=="__main__":
    main()