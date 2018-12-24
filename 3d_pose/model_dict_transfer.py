#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:48:37 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""
import os

import torch

from src.configer import Configer
from src.model import LinearModel as l1
from src.model1 import LinearModel as l2

def save_ckpt(state, ckpt_path, is_best=True):
    if is_best:
        file_path = os.path.join(ckpt_path, 'gt_ckpt_best.pth.tar')
        torch.save(state, file_path)
    else:
        file_path = os.path.join(ckpt_path, 'gt_ckpt_last.pth.tar')
        torch.save(state, file_path)

if __name__=='__main__':
    cfg = Configer('3d_pose_baseline.cfg')
    model_new = l1(cfg)
    model_new_dict = model_new.state_dict()

    model_old = l2()
    model_old_dict = model_old.state_dict()

    print(">>> loading ckpt from '{}'".format('/backup/gt_ckpt_best_old.pth.tar'))
    ckpt = torch.load('./backup/gt_ckpt_best_old.pth.tar')
    model_old.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print(">>> transfering old to new")
    
    for old,new in zip(model_old_dict, model_new_dict):        
        model_new_dict[new] = model_old_dict[old]
    print("Done!!!")
    model_new.load_state_dict(model_new_dict)
    model_new_dict = model_new.state_dict()
    
    for old,new in zip(model_old_dict, model_new_dict):
        print(old)
        #print(model_old_dict[old])
        print(new)
        #print(model_new_dict[new])
        print('\n')
    
    '''
    save_ckpt({'epoch': ckpt['epoch'],
               'lr': ckpt['lr'],
               'step': ckpt['step'],
               'err': ckpt['err'],
               'state_dict': model_new.state_dict(),
               'optimizer': ckpt['optimizer']},
              ckpt_path='./backup',
              is_best=True)
    '''
    
    inp = torch.rand(2, 32)
    
    model_old.cuda()
    model_new.cuda()
    model_old.eval()
    model_new.eval()
    
    from torch.autograd import Variable
    inputs = Variable(inp.cuda())
    
    #print(model_old_dict['linear_stages.0.w2.weight'])
    #print(model_new_dict['linearblocks.0.linear2.weight'])
    #print(model_old_dict['linear_stages.0.w2.bias'])
    #print(model_new_dict['linearblocks.0.linear2.bias'])
    outputs_old = model_old(inputs)
    outputs_new = model_new(inputs)
    count=0
    for a,b in zip(outputs_old[0],outputs_new[0]):
        if a!=b:
            print(a)
            print(b)
            print(count)
        count+=1
    
    print(outputs_old[0])
    print(outputs_new[0])
    