#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:00:43 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""
import torch
import torch.nn as nn

import numpy as np

from src.model import LinearModel
#from src.model1 import LinearModel as l1
from src.train import Trainer
from src.test import Tester
from src.configer import Configer

actions = ["Directions",
               "Discussion",
               "Eating",
               "Greeting",
               "Phoning",
               "Photo",
               "Posing",
               "Purchases",
               "Sitting",
               "SittingDown",
               "Smoking",
               "Waiting",
               "WalkDog",
               "Walking",
               "WalkTogether"]

if __name__ == '__main__':
    cfg = Configer('3d_pose_baseline.cfg')
    
    lr_now=cfg.get_learning_rate()
    
    model = LinearModel(cfg)
    model = model.cuda()
    
    glob_step = 0
    start_epoch = 0
    err_best = 1000
    # load ckpt
    if cfg.is_train()!=1:
        ckpt = torch.load(cfg.get_ckpt())#'./backup/gt_ckpt_best_old.pth.tar'
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    else:
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
        model.apply(weight_init)

    tester = Tester(model, cfg, actions)
    if cfg.is_train(): 
        op_ckpt=None
        if cfg.is_train==2:
            op_ckpt=ckpt['optimizer']
        
        trainer = Trainer(model, cfg, actions, op_ckpt)
        for epoch in range(start_epoch, cfg.get_epochs()):
            print('==========================')
            print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
            glob_step, lr_now, loss_train, optimizer=trainer(glob_step, epoch)

            loss_test, err_test = tester.validating()
            
            # save ckpt
            is_best = err_test < err_best
            err_best = min(err_test, err_best)
            if is_best:
                trainer.save_ckpt({'epoch': epoch + 1,
                                'lr': lr_now,
                                'step': glob_step,
                                'err': err_best,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                               ckpt_path='./backup',
                               is_best=True)
            else:
                trainer.save_ckpt({'epoch': epoch + 1,
                                'lr': lr_now,
                                'step': glob_step,
                                'err': err_best,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                               ckpt_path='./backup',
                               is_best=False)
            
    else:
        err_set = []
        for action in actions:
            err_test=tester.tesing(action)
            err_set.append(err_test)
        print(">>>>>> TEST results:")
        for action in actions:
            print("{}".format(action), end='\t')
        print("\n")
        for err in err_set:
            print("{:.4f}".format(err), end='\t')
        print(">>>\nERRORS: {}".format(np.array(err_set).mean()))
