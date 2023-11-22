import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
# import lowlight_model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import argparse
import time
import torch
import torchvision.transforms as tfs
from niqe import niqe
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models_x import *
from datasets import *


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to load the saved checkpoint")
parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--model_dir", type=str, default="LUTs/unpaired/cityscapes", help="directory of saved models")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path)
	data_lowlight = (np.asarray(data_lowlight)/255.0)

	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	# SCL_LLE_net = lowlight_model.enhance_net_nopool().cuda()
	# SCL_LLE_net.load_state_dict(torch.load('checkpoints/SCL-LLE.pth'))
	trilinear_ = TrilinearInterpolation()
	LUT0 = Generator3DLUT_identity()
	LUT1 = Generator3DLUT_zero()
	LUT2 = Generator3DLUT_zero()
	LUTs = torch.load("saved_models/%s/LUTs_0.pth" % (opt.model_dir))
	LUT0.load_state_dict(LUTs["0"])
	LUT1.load_state_dict(LUTs["1"])
	LUT2.load_state_dict(LUTs["2"])
	classifier = Classifier()
	classifier.load_state_dict(torch.load("saved_models/%s/classifier_0.pth" % (opt.model_dir)))
	if cuda:
		LUT0 = LUT0.cuda()
		LUT1 = LUT1.cuda()
		LUT2 = LUT2.cuda()
		#LUT3 = LUT3.cuda()
		#LUT4 = LUT4.cuda()
		classifier = classifier.cuda()
	start = time.time()
	# _,enhanced_image,_ = SCL_LLE_net(data_lowlight)
	pred = classifier(data_lowlight).squeeze()
	LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT 
	_,combine_A = trilinear_(LUT,data_lowlight)
	end_time = (time.time() - start)
	print(end_time)
	# t = np.array(Image.open(pre_imgs).convert('LA'))[:,:,0] 
	niqescore = niqe(combine_A)
	# niqes.append(niqescore)
	# niqes_mean = np.mean(niqes)
	print(f'\rNIQE:{niqescore:.5f}')
	# image_path = image_path.replace('test_data','result')
	# result_path = image_path
	# if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
	# 	os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	# torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
	with torch.no_grad():
		filePath = '/home2/zyy/Image-Adaptive-3DLUT/data/DICM'
	
		file_list = os.listdir(filePath)

		for file_name in file_list:
			# test_list = glob.glob(filePath+file_name+"/*") 
			# for image in test_list:
			print(file_name)
			lowlight(file_name)

		

