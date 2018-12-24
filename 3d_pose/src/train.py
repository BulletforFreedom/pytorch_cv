#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:59:11 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""

import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src import utils
from src.data_progress.human36m import Human36M


class Trainer():
    def __init__(self, model, config, actions, op_ckpt=None):
        self.model = model
        self.cfg = config
        self.lr_now = self.cfg.get_learning_rate()
        
        self.loss_fuction = nn.MSELoss(size_average=True).cuda()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr_now)
        if op_ckpt:
            self.optimizer.load_state_dict(op_ckpt)

        # load dadasets for training
        self.train_loader = DataLoader(
            dataset=Human36M(actions=actions, data_path=self.cfg.get_data_path(), use_hg=False),
            batch_size=self.cfg.get_batch_size(),
            shuffle=True,
            num_workers=self.cfg.get_num_workers(),
            pin_memory=True)
        print(">>> trainng data loaded !")

    def save_ckpt(self, state, ckpt_path, is_best=True):
        if is_best:
            file_path = os.path.join(ckpt_path, 'ckpt_best.pth.tar')
            torch.save(state, file_path)
        else:
            file_path = os.path.join(ckpt_path, 'ckpt_last.pth.tar')
            torch.save(state, file_path)

    def _lr_decay(self, step, lr, decay_step, gamma):
        lr = lr * gamma ** (step / decay_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def __call__(self, glob_step, epoch):
        losses = utils.AverageMeter()

        self.model.train()
        
        start = time.time()
        batch_time = 0

        for i, (inps, tars) in enumerate(self.train_loader):
            glob_step += 1
            if glob_step % self.cfg.get_lr_decay() == 0 or glob_step == 1:
                self.lr_now = self._lr_decay( 
                        glob_step, 
                        self.cfg.get_learning_rate(), 
                        self.cfg.get_lr_decay(), 
                        self.cfg.get_lr_gamma())
            inputs = Variable(inps.cuda())
            targets = Variable(tars.cuda(async=True))

            outputs = self.model(inputs)

            # calculate loss
            self.optimizer.zero_grad()
            loss = self.loss_fuction(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            loss.backward()
            #losses.update(loss.data[0], inputs.size(0))

            if True:
                '''
                Max norm constraints. 
                Another form of regularization is to enforce an absolute upper bound 
                on the magnitude of the weight vector for every neuron and 
                use projected gradient descent to enforce the constraint. 
                In practice, this corresponds to performing the parameter update as normal,
                and then enforcing the constraint by clamping the weight vector w⃗  
                of every neuron to satisfy ∥w⃗ ∥2<c. 
                Typical values of c are on orders of 3 or 4. 
                Some people report improvements when using this form of regularization. 
                One of its appealing properties is that network cannot “explode” even 
                when the learning rates are set too high because the updates are always bounded.
                '''
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            
            # update summary
            if (i + 1) % 100 == 0:
                batch_time = time.time() - start
                start = time.time()
            print('Epoch: {epoch}/{epochs} | ({batch}/{size}) | batch: {batchtime:.4}ms | loss: {loss:.4f}' \
            .format(epoch=epoch+1,
                    epochs=self.cfg.get_epochs(),
                    batch=i + 1,
                    size=len(self.train_loader),
                    batchtime=batch_time * 10.0,
                    loss=losses.avg))
            
        return glob_step, self.lr_now, losses.avg, self.optimizer
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!lr_now 