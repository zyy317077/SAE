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

from models import *
from datasets import *


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to load the saved checkpoint")
parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
parser.add_argument("--model_dir", type=str, default="LUTs/", help="directory of saved models")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'

	data_lowlight = torch.from_numpy(image_path).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	trilinear_ = TrilinearInterpolation()
	LUT0 = Generator3DLUT_identity()
	LUT1 = Generator3DLUT_zero()
	LUT2 = Generator3DLUT_zero()
	LUTs = torch.load("saved_models/%s/LUTs.pth" % (opt.model_dir))
	LUT0.load_state_dict(LUTs["0"])
	LUT1.load_state_dict(LUTs["1"])
	LUT2.load_state_dict(LUTs["2"])
	classifier = Classifier()
	classifier.load_state_dict(torch.load("saved_models/%s/classifier.pth" % (opt.model_dir)))
	if cuda:
		LUT0 = LUT0.cuda()
		LUT1 = LUT1.cuda()
		LUT2 = LUT2.cuda()

		classifier = classifier.cuda()
	start = time.time()

	pred = classifier(data_lowlight).squeeze()
	LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT 
	_,combine_A = trilinear_(LUT,data_lowlight)
	end_time = (time.time() - start)
	print(end_time)

	niqescore = niqe(combine_A)

	print(f'\rNIQE:{niqescore:.5f}')


if __name__ == '__main__':
	with torch.no_grad():
		filePath = './data/DICM'
	
		file_list = os.listdir(filePath)

		for file_name in file_list:
			# test_list = glob.glob(filePath+file_name+"/*") 
			# for image in test_list:
			print(file_name)
			lowlight(file_name)

		

