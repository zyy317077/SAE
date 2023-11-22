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
parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--model_dir", type=str, default="checkpoint2", help="directory of saved models")
parser.add_argument("--coarse_dir", type=str, default='./save_images/syn/coarse',help="path to Dataset")
parser.add_argument("--fine_dir", type=str, default='./save_images/LIME',help="path to Dataset")
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
seg_net = ERFNet(19)
net = u_net_heavy(6,3)
# net = nn.DataParallel(net)
# model = Network()
trilinear_ = TrilinearInterpolation() 

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    seg_net = seg_net.cuda()
    net = net.cuda()
    # model = model.cuda()
    # criterion_pixelwise.cuda()

def generator(img):
    pred,out = net(img, None, None, True)#[1,128,64,64]->[1,3,1,1],[1,19,360,360]
    pred = pred.squeeze()
        
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT #+ pred[3] * LUT3.LUT + pred[4] * LUT4.LUT

    combine_A = img.new(img.size())
    _, combine_A = trilinear_(LUT, img)
    return combine_A,out

def test():
    seg_net.load_state_dict(torch.load("saved_models/%s/segNet_%d.pth" % (opt.model_dir, opt.epoch)), strict=False)
    seg_net.eval()  
    net.load_state_dict(torch.load("saved_models/%s/model_%d.pth" % (opt.model_dir, opt.epoch)), strict=False)
    net.eval()
    # model.load_state_dict(torch.load('./initmodel/upe.pt'))    
    LUTs = torch.load("saved_models/%s/LUTs_%d.pth" % (opt.model_dir,opt.epoch))
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    LUT0.eval()
    LUT1.eval()
    LUT2.eval()
    
    with torch.no_grad():
    # """Saves a generated sample from the validation set"""
        test_path = '/home2/zyy/Image-Adaptive-3DLUT/dataset/syn/test/low'
        target_path = '/home2/zyy/Image-Adaptive-3DLUT/dataset/syn/test/high'
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
            
            if test_img.size[0] < 512 or test_img.size[1] < 512:
                test_img = transforms.Pad(padding=10,padding_mode='edge')(test_img)
                target_img = transforms.Pad(padding=10,padding_mode='edge')(target_img)
            i, j, h, w = transforms.RandomCrop.get_params(test_img, output_size=(512,512))
            test_img = TF.crop(test_img, i, j, h, w)
            target_img = TF.crop(target_img, i, j, h, w)
            test_img = tfs.ToTensor()(test_img)
            target_img = tfs.ToTensor()(target_img)
            test_img = test_img.reshape(1,test_img.shape[0],test_img.shape[1],test_img.shape[2])
            target_img = target_img.reshape(1,target_img.shape[0],target_img.shape[1],target_img.shape[2])
            test_img = test_img.cuda()    
            target_img = target_img.cuda() 

            # ----------
            #  model
            # ----------
            pred,seg_pred = seg_net(test_img)#[1,128,64,64]->[1,3,1,1],[1,19,360,360]
            pred = pred.squeeze()
            LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT #+ pred[3] * LUT3.LUT + pred[4] * LUT4.LUT
            _, coarse_pred = trilinear_(LUT, test_img)
            seg_gray = seg_pred.argmax(dim=1)           #[1,512,512]   #torch.argmax返回指定维度最大值的序号
            seg_gray = seg_gray.unsqueeze(1).float() 
            refined_pred = net(test_img, coarse_pred, seg_gray)
            # save_image(coarse_pred, os.path.join(opt.coarse_dir,"%s.png" % (test_name[:-4])), nrow=1, normalize=False)
            # save_image(refined_pred, os.path.join(opt.fine_dir,"%s.bmp" % (test_name[:-4])), nrow=1, normalize=False)

            # ssim1 = ssim(test_img_tensor, target_img_tensor).item()
            ssim1 = ssim(refined_pred, target_img).item()
            psnr1 = psnr(refined_pred, target_img)
            ssims.append(ssim1)
            psnrs.append(psnr1)
        ssim_eval = np.mean(ssims)
        psnr_eval = np.mean(psnrs)

        print(f'\n ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')
            # t = np.array(Image.open(refined_pred).convert('LA'))[:,:,0] 
            # niqescore = niqe(t)
            # niqes.append(niqescore)
        # niqes_mean = np.mean(niqes)
        # print(f'\rNIQE:{niqes_mean:.5f}')


    # os.makedirs(out_dir, exist_ok=True)
    # for i, batch in enumerate(dataloader):
    #     real_A = Variable(batch["A_input"].type(Tensor))
    #     img_name = batch["input_name"]
    #     fake_B = generator(real_A)
        
    #     #real_B = Variable(batch["A_exptC"].type(Tensor))
    #     #img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -1)
    #     #save_image(img_sample, "images/LUTs/paired/JPGsRGB8_to_JPGsRGB8_WB_original_5LUT/%s.png" % (img_name[0][:-4]), nrow=3, normalize=False)
    #     save_image(fake_B, os.path.join(out_dir,"%s.png" % (img_name[0][:-4])), nrow=1, normalize=False)

def test_speed():
    t_list = []
    for i in range(1,10):
        img_input = Image.open(os.path.join("./data/fiveK/input/JPG","original","a000%d.jpg"%i))
        img_input = torch.unsqueeze(TF.to_tensor(TF.resize(img_input,(4000,6000))),0)
        real_A = Variable(img_input.type(Tensor))
        torch.cuda.synchronize()
        t0 = time.time()
        for j in range(0,100):
            fake_B = generator(real_A)
        
        torch.cuda.synchronize()
        t1 = time.time()
        t_list.append(t1 - t0)
        print((t1 - t0))
    print(t_list)

# ----------
#  evaluation
# ----------
test()

#test_speed()
