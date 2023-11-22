import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class Cityscapes(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    # def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
    def __init__(self, root, train, mode1 = 'H', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        # self.mode = 'gtFine'
        self.mode1 = 'H'
        self.target_type = target_type
        # self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.images_dir1 = os.path.join(self.root, 'low')       
        self.targets_dir1 = os.path.join(self.root, 'high')
        # self.images_dir2 = os.path.join(self.root, 'test_L')       
        # self.targets_dir2 = os.path.join(self.root, 'test_H')
        # self.semantics_dir = os.path.join(self.root, 'train_M')
        self.transform = transform

        # self.split = split
        self.images1 = []
        self.targets1 = []
        # self.images2 = []
        # self.targets2 = []
        # self.semantics = []

        # if split not in ['train', 'test', 'val']:
        #     raise ValueError('Invalid split for mode! Please use split="train", split="test"'
        #                      ' or split="val"')

        if not os.path.isdir(self.images_dir1) or not os.path.isdir(self.targets_dir1) :
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        # for city in os.listdir(self.images_dir):
        #     img_dir = os.path.join(self.images_dir, city)
        #     target_dir = os.path.join(self.targets_dir, city)

        for file_name in os.listdir(self.images_dir1):
            self.images1.append(os.path.join(self.images_dir1, file_name))            
            target_name = '{}_{}.png'.format(file_name.split('_L')[0],
                                              self.mode1)
                                            # self._get_target_suffix(self.mode, self.target_type))
            self.targets1.append(os.path.join(self.targets_dir1, target_name))
        # for file_name in os.listdir(self.images_dir2):
        #     self.images2.append(os.path.join(self.images_dir2, file_name))            
        #     target_name = '{}_{}.png'.format(file_name.split('_L')[0],
        #                                     self.mode1)
        #                                     # self._get_target_suffix(self.mode, self.target_type))
        #     self.targets2.append(os.path.join(self.targets_dir1, target_name))
    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        # if self.train == True:
        image = Image.open(self.images1[index])
        # image = Image.open(self.images[index])
        target = Image.open(self.targets1[index])
        # else:
        #     image = Image.open(self.images2[index])
        #     # image = Image.open(self.images[index])
        #     target = Image.open(self.targets2[index])
        # semantic = Image.open(self.semantics[index])
        # if self.transform:
        #     image, target = self.transform(image, target)
        # semantic = self.encode_target(semantic)
        # ratio_H = np.random.uniform(0.6,0.8)
        # ratio_W = np.random.uniform(0.3,0.5)
        # # W,H = image._size
        # image = transforms.Resize([512, 1024])
        # target = transforms.Resize([512, 1024])
        # crop_h = round(H*ratio_H)
        # crop_w = round(W*ratio_W)
        image = transforms.Pad(padding=4,padding_mode='edge')(image)
        target = transforms.Pad(padding=4,padding_mode='edge')(target)
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(512,512))
        # image = TF.resized_crop(image, i, j, h, w)
        # target = TF.resized_crop(target, i, j, h, w)
        image = TF.crop(image, i, j, h, w)
        target = TF.crop(target, i, j, h, w)
        # semantic = TF.crop(semantic, i, j, h, w)

        image,target = self.augData(image.convert('RGB'),target.convert('RGB'))
        # if self.transform:
        #     # image, target = self.transform(image, target)
        # target = self.encode_target(target)
        image = TF.to_tensor(image)
        target = TF.to_tensor(target) 
        # semantic = TF.to_tensor(semantic) 

        return image, target, 
    def augData(self, data, target1):
        if self.train:
            rand_hor = np.random.randint(0, 1)
            rand_rot = np.random.randint(0, 3)
            data = transforms.RandomHorizontalFlip(rand_hor)(data)
            target1 = transforms.RandomHorizontalFlip(rand_hor)(target1)
            # target2 = transforms.RandomHorizontalFlip(rand_hor)(target2)
            if rand_rot:
                data = TF.rotate(data, 90*rand_rot)
                target1 = TF.rotate(target1, 90*rand_rot)
                # target2 = TF.rotate(target2, 90*rand_rot)
        
        return data, target1  

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)