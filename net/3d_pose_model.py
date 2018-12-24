#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:59:53 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""

import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, linearsize, dropout):
        super(LinearBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(linearsize, linearsize)
        self.bn1 = nn.BatchNorm1d(linearsize)

        self.linear2 = nn.Linear(linearsize, linearsize)
        self.bn2 = nn.BatchNorm1d(linearsize)
               

    def forward(self, x):
        
        y = self.linear1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.linear2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class LinearModel(nn.Module):
    def __init__(self, config):
        super(LinearModel, self).__init__()
        self.cfg=config
        self.net_strucure=self.cfg.get_blocks()
        self.linear_size=int(self.net_strucure['linearsize'])
        self.num_linear_blocks=int(self.net_strucure['num_linear_blocks'])
        self.drop_out=float(self.net_strucure['drop_out'])

        self.linear1 = nn.Linear(self.cfg.get_final_inp_dim(), self.linear_size)
        self.batch_norm = nn.BatchNorm1d(self.linear_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.drop_out)

        self.linearblocks = []
        for i in range(self.num_linear_blocks):
            self.linearblocks.append(LinearBlock(self.linear_size, self.drop_out))
        self.linearblocks = nn.ModuleList(self.linearblocks)

        self.linear2 = nn.Linear(self.linear_size, self.cfg.get_out_dim())

    def forward(self, input):
        output = self.linear1(input)
        output = self.batch_norm(output)
        output = self.relu(output)
        output = self.dropout(output)
        
        for i in range(self.num_linear_blocks):
            output = self.linearblocks[i](output)
        
        output = self.linear2(output)
        return output


if __name__ == '__main__':
    from src.configer import Configer
    cfg = Configer('3d_pose_baseline.cfg')

    inp = torch.rand(2, 32)

    model = LinearModel(cfg)
    model = model.cuda()

    # load ckpt
    if cfg.is_train():
        ckpt = torch.load(cfg.get_ckpt())
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
    model.eval()

    from torch.autograd import Variable
    inputs = Variable(inp.cuda())

    outputs = model(inputs)
    print(outputs)
