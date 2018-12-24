#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:01:12 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from torch.utils import data

import data_loader.transforms as trans
import data_augmentation.aug_transforms as aug_trans
from data_loader.res_data_loader import ResDataLoader


class ClstDataLoader(object):

    def __init__(self, configer, is_debug=False):
        self.configer = configer
        self.is_debug = is_debug
        self.aug_train_transform = aug_trans.AugCompose(self.configer)        
        self.img_transform = trans.Compose()
        self.img_transform.add(trans.ResizeImage(configer))
        self.img_transform.add(trans.ToTensor())
        #self.img_transform.add(trans.Normalize(self.configer.get_dataset_mean(),
                                               #self.configer.get_dataset_std()))
        
    def get_train_loader(self):
        bs=self.configer.get_batch_size()
        
        loader = data.DataLoader(
            ResDataLoader(root_dir=os.path.join(self.configer.get_data_path(), 
                                                'train.txt'),
                          classes=self.configer.get_num_classes(),
                         aug_transform=self.aug_train_transform,
                         img_transform=self.img_transform),
            batch_size=bs, shuffle=True,
            num_workers=self.configer.get_num_workers(), 
            pin_memory=True)
        return loader
    
    def get_val_loader(self):
        bs=self.configer.get_batch_size()
        
        loader = data.DataLoader(
            ResDataLoader(root_dir=os.path.join(self.configer.get_data_path(),
                                                'val.txt'),
                          classes=self.configer.get_num_classes(),
                         aug_transform=None,
                         img_transform=self.img_transform),
            batch_size=bs, shuffle=True,
            num_workers=self.configer.get_num_workers(), 
            pin_memory=True)
        return loader