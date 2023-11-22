import numpy as np
import torch
from torch.autograd import Variable
import glob
import cv2
from PIL import Image as PILImage
# import Model as Net
from models.model_UNet import *
import os
import time
from argparse import ArgumentParser
import torchvision.transforms as tfs
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


pallete = [[128, 64, 128],
           [244, 35, 232],
           [70, 70, 70],
           [102, 102, 156],
           [190, 153, 153],
           [153, 153, 153],
           [250, 170, 30],
           [220, 220, 0],
           [107, 142, 35],
           [152, 251, 152],
           [70, 130, 180],
           [220, 20, 60],
           [255, 0, 0],
           [0, 0, 142],
           [0, 0, 70],
           [0, 60, 100],
           [0, 80, 100],
           [0, 0, 230],
           [119, 11, 32],
           [0, 0, 0]]

def evaluateModel(args, model,img):
    # gloabl mean and std values
    mean = [72.3923111, 82.90893555, 73.15840149]
    std = [45.3192215, 46.15289307, 44.91483307]

    # for i, imgName in enumerate(image_list):
    #     img = cv2.imread(imgName)
    #     if args.overlay:
    #         img_orig = np.copy(img)

    img = img.astype(np.float32)
    for j in range(3):
        img[:, :, j] -= mean[j]
    for j in range(3):
        img[:, :, j] /= std[j]

    # resize the image to 1024x512x3
    # img = cv2.resize(img, (1024, 512))
    # if args.overlay:
    #     img_orig = cv2.resize(img_orig, (1024, 512))

    # img /= 255
    # img = img.transpose((2, 0, 1))
    # img_tensor = torch.from_numpy(img)
    # img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
    # img_variable = Variable(img_tensor, volatile=True)
    # if args.gpu:
    #     img_variable = img_variable.cuda()
    img_out = model(img)
    img_notensor = torch.squeeze(img, 0) 

    classMap_numpy = img_out[0].max(0)[1].byte().cpu().data.numpy()

    name = img.split('/')[-1]

    if args.colored:
        classMap_numpy_color = np.zeros((img_notensor.shape[1], img_notensor.shape[2], img_notensor.shape[0]), dtype=np.uint8)
        for idx in range(len(pallete)):
            [r, g, b] = pallete[idx]
            classMap_numpy_color[classMap_numpy == idx] = [b, g, r]
        cv2.imwrite(args.savedir + os.sep + 'c_' + name.replace(args.img_extn, 'png'), classMap_numpy_color)

    # cv2.imwrite(args.savedir + os.sep + name.replace(args.img_extn, 'png'), classMap_numpy)


def main(args):
    # if args.modelType == 1:
    # modelA = Net()  # Net.Mobile_SegNetDilatedIA_C_stage1(20)
    # model_weight_file = args.weightsDir + os.sep + 'encoder' + os.sep + 'espnet_p_' + str(p) + '_q_' + str(
    #     q) + '.pth'
    net = LHSNet()
    net.load_state_dict(torch.load("saved_models/%s/model_%d.pth" % (args.model_dir, args.epoch)), strict=False)
    if args.gpu:
        net = net.cuda()
    # set to evaluation mode
    net.eval()
    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)
    with torch.no_grad():
        test_path = '/home2/zyy/Image-Adaptive-3DLUT/dataset/syn/test/low'
        # target_path = '/home2/zyy/Image-Adaptive-3DLUT/data/syn/test/high'
        test_imgs_dir = sorted(os.listdir(test_path))
        # target_imgs_dir = sorted(os.listdir(target_path))
        for i in range(len(test_imgs_dir)):
            test_name = test_imgs_dir[i]
            # target_name = target_imgs_dir[i]
            test_imgs = os.path.join(test_path, test_name)
            # target_imgs = os.path.join(target_path, target_name)
            test_img = Image.open(test_imgs).convert('RGB')
            # target_img = Image.open(target_imgs) 
            # test_img = tfs.Resize([512, 512])(test_img)
            # target_img = tfs.Resize([512, 512])(target_img)
            
            if test_img.size[0] < 512 or test_img.size[1] < 512:
                test_img = transforms.Pad(padding=10,padding_mode='edge')(test_img)
                # target_img = transforms.Pad(padding=10,padding_mode='edge')(target_img)
            i, j, h, w = transforms.RandomCrop.get_params(test_img, output_size=(512,512))
            test_img = TF.crop(test_img, i, j, h, w)
            # target_img = TF.crop(target_img, i, j, h, w)

            test_img = tfs.ToTensor()(test_img)
            # target_img = tfs.ToTensor()(target_img)
            test_img = test_img.reshape(1,test_img.shape[0],test_img.shape[1],test_img.shape[2])
            # target_img = target_img.reshape(1,target_img.shape[0],target_img.shape[1],target_img.shape[2])
            test_img = test_img.cuda()    
            # target_img = target_img.cuda() 
            # coarse_pred,seg_pred = generator(test_img)
            _,seg_pred = net(test_img, None, None, True)
            
            evaluateModel(args, net, seg_pred)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNet", help='Model name')
    parser.add_argument('--data_dir', default="./data", help='Data directory')
    parser.add_argument('--img_extn', default="png", help='RGB Image format')
    parser.add_argument("--epoch", type=int, default=0, help="epoch to load the saved checkpoint")
    parser.add_argument("--model_dir", type=str, default="checkpoint0", help="directory of saved models")
    parser.add_argument('--savedir', default='./save_images/semantic', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=bool, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weightsDir', default='./pretrained/', help='Pretrained weights directory.')
    parser.add_argument('--colored', default=True, type=bool, help='If you want to visualize the '
                                                                   'segmentation masks in color')
    parser.add_argument('--classes', default=20, type=int, help='Number of classes in the dataset. 20 for Cityscapes')

    args = parser.parse_args()
    # assert (args.modelType == 1) and args.decoder, 'Model type should be 2 for ESPNet-C and 1 for ESPNet'
    # if args.overlay:
    #     args.colored = True # This has to be true if you want to overlay
    main(args)
