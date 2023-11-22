import json
import os
from collections import namedtuple
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
# from image_adaptive_lut_train_paired import opt
import torchvision.transforms as transforms
import torchvision.transforms as tfs
import torchvision.transforms.functional as TF


class Cityscapes(data.Dataset):

    def __init__(self, root, train, size, target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.size = size
        self.target_type = target_type
        self.images_dir1 = os.path.join(self.root, 'low')       
        self.targets_dir1 = os.path.join(self.root, 'high')
        self.transform = transform

        self.images1 = []
        self.targets1 = []

        if not os.path.isdir(self.images_dir1) or not os.path.isdir(self.targets_dir1) :
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for file_name in os.listdir(self.images_dir1):
            self.images1.append(os.path.join(self.images_dir1, file_name))            
            self.targets1.append(os.path.join(self.targets_dir1, file_name))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image_name = self.images1[index]
        image = Image.open(self.images1[index]).convert("RGB")
        target = Image.open(self.targets1[index]).convert("RGB")

        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.size, self.size))
            image = TF.crop(image, i, j, h, w)
            target = TF.crop(target, i, j, h, w)

        else:
            image = tfs.Resize([512, 512])(image)
            target = tfs.Resize([512, 512])(target)


        image = TF.to_tensor(image)
        target = TF.to_tensor(target) 
        return image, target, image_name

    def augData(self, data, target1):
        if self.train:
            rand_hor = np.random.randint(0, 1)
            rand_rot = np.random.randint(0, 3)
            data = transforms.RandomHorizontalFlip(rand_hor)(data)
            target1 = transforms.RandomHorizontalFlip(rand_hor)(target1)
            if rand_rot:
                data = TF.rotate(data, 90*rand_rot)
                target1 = TF.rotate(target1, 90*rand_rot)
        
        return data, target1  

    def __len__(self):
        return len(self.images1)

