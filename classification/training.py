#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:47:31 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from classification import utils

class Training(object):
    def __init__(self, cfg, net, trainloader):
        self.cfg = cfg
        self.net = net
        self.trainloader=trainloader
        self.batch_len=len(self.trainloader)
        self.lr_now = self.cfg.get_learning_rate()
        self.criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
        self.optimizer = optim.SGD(self.net.parameters(),
                                   lr=self.lr_now, 
                                   momentum=0.9, 
                                   weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
    
    def __call__(self, epoch):
        # 定义损失函数和优化方式
        loss=0
        losses = utils.AverageMeter()
        #with open("log.txt", "w")as f2:
        for i, (inputs, labels, image_dirs) in enumerate(self.trainloader):
            start = time.time()
            # 准备数据
            glob_step=epoch*self.batch_len+i+1
            if glob_step % 1 == 0:
                self.lr_now = self._lr_decay( 
                        glob_step, 
                        self.cfg.get_learning_rate(), 
                        self.cfg.get_lr_decay(), 
                        self.cfg.get_lr_gamma())
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            self.optimizer.zero_grad()
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda(async=True))
            # forward + backward
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            losses.update(loss.item(), inputs.size(0))
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
            self.optimizer.step()
            
            # 每训练1个batch打印一次loss和准确率
            batch_time = time.time() - start
            print('Epoch: {epoch}/{epochs} | learning_rate: {lr:.6f} | loss: {loss:.4f} | batch: {batchtime:.4}ms | ({batch}/{length})' \
            .format(epoch=epoch+1,
                    epochs=self.cfg.get_epochs(),
                    batch=i + 1,
                    length=self.batch_len,
                    lr=self.lr_now,
                    batchtime=batch_time * 10.0,
                    loss=losses.avg))
        #self.scheduler.step()
        return self.lr_now, loss, self.optimizer
    
    def _lr_decay(self, step, lr, decay_step, gamma):
        lr = lr * gamma ** (step / decay_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr