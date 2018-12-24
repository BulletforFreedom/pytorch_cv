#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 14:40:31 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#from PIL import Image
import cv2
import numpy as np
import torch
import torch.utils.data as data

#from datasets.det.det_data_utilizer import DetDataUtilizer
#from utils.helpers.image_helper import ImageHelper


class ResDataLoader(data.Dataset):

    def __init__(self, root_dir=None, classes=4, aug_transform=None,
                 img_transform=None):
        super(ResDataLoader, self).__init__()
        self.img_list, self.label_list = self.__list_dirs(root_dir, classes)
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        #self.det_data_utilizer = DetDataUtilizer(configer)
    
    def __getitem__(self, index):
        #img = Image.open(self.img_list[index]).convert('RGB')
        img = cv2.imread(self.img_list[index])
        if self.aug_transform:
            img = self.aug_transform(img)     
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          
        img = img[:,:,:]/255.0
        if self.img_transform is not None:
            img = self.img_transform(img)
            
        label=self.label_list[index]
        label = torch.from_numpy(np.array(label)).long()

        return img, label, self.img_list[index]
    

    def __len__(self):

        return len(self.img_list)

    def __list_dirs(self, root_dir, classes):
        label_list = list()
        img_list = list()
        txt_path = root_dir
        f_r = open(txt_path)
        data = f_r.readlines()
        f_r.close()

        for line in data:
            line = line.split('$')
            img_list.append(line[0])
            #label=[ 0 for i in range(classes)]
            #label[int(line[1])]=1
            label_list.append(int(line[1]))
            if not os.path.exists(line[0]):
                print('Image Path: {} not exists.'.format(line[0]))
                exit(1)

        return img_list, label_list