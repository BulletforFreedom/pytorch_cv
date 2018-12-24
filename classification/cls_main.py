#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 17:26:01 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""
import os, sys
import glob
import cv2

import torchvision.models as models
import torch
import torch.nn as nn

sys.path.append('/home/lsk/Downloads/pytorch_cv/project')
from data_loader.cls_data_loader import ClstDataLoader as DataLoader
from cfg.configer import Configer
from classification.training import Training
from classification.testing import Testing
from classification.validating import Validating
from common.util import Util

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 训练
if __name__ == "__main__":
    cfg = Configer("resnet34.cfg")
    net = models.resnet34(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, cfg.get_num_classes())
    
    start_epoch=0
    best_acc = 0  #2 初始化best test accuracy
    if cfg.is_train()!=1:
        ckpt = torch.load(cfg.get_ckpt())#'./backup/gt_ckpt_best_old.pth.tar'
        start_epoch = ckpt['epoch']
        best_acc = ckpt['acc']
        net.load_state_dict(ckpt['state_dict'])
        print(">>> ckpt loaded (epoch: {} | acc: {})".format(start_epoch, best_acc))
    
    net = net.cuda()
    
    if cfg.is_train():
        dataloader = DataLoader(cfg)
        trainloader = dataloader.get_train_loader()
        valloader = dataloader.get_val_loader()
        
        trainer=Training(cfg, net, trainloader)
        validater=Validating(cfg, net, valloader)
        
        print("Start Training, Resnet-34!")  # 定义遍历数据集的次数
        for epoch in range(start_epoch, cfg.get_epochs()):
            print('==========================')              
            sum_loss = 0.0
            correct = 0.0
            total = 0.0  
            
            net.train()
            lr_now, loss_train, optimizer=trainer(epoch)
            # 每训练完一个epoch测试一下准确率
            print("Waiting Test!")
            net.eval()
            val_acc=validater()
            # save ckpt
            is_best = val_acc > best_acc
            
            best_acc = max(val_acc, best_acc)            
            if is_best:
                best_epoch=epoch
                Util.save_ckpt(cfg, {'epoch': epoch + 1,
                                'lr': lr_now,
                                'loss': loss_train,
                                'acc': best_acc,
                                'state_dict': net.state_dict(),
                                'optimizer': optimizer.state_dict()})
            
        print("Training Finished, TotalEPOCH=%d" % cfg.get_epochs())
    else:
        data_paths=glob.glob(os.path.join(cfg.get_data_dir(),
                                               cfg.get_project_name(),
                                               'test/*.jpg'))
        name_seq = cfg.get_dataset_name_seq() 
        tester=Testing(cfg, net)        
        net.eval()
        
        for path in data_paths:
            print('Processing %s' %path.split('/')[-1])
            images=cv2.imread(path)
            predicted=tester(images)
            cv2.imshow(name_seq[predicted],images)
            #cv2.moveWindow(name,10,10)
            k = cv2.waitKey(0)
            if k==ord('q'):
                cv2.destroyAllWindows()
                break   
            else:
                cv2.destroyWindow(name_seq[predicted])