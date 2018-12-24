#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:00:05 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np

from src import utils
from src.data_progress.human36m import Human36M
from src.procrustes import get_transformation

class Tester(object):
    def __init__(self, model, config, actions):
        self.model = model
        self.cfg = config
        self.loss_fuction = nn.MSELoss(size_average=True).cuda()
        # load statistics data
        self.stat_3d = torch.load(os.path.join(self.cfg.get_data_path(), 'stat_3d.pth.tar'))
        # load dadasets for validating
        if self.cfg.is_train:
            self.data_loader = DataLoader(
                dataset=Human36M(actions=actions, data_path=self.cfg.get_data_path(), use_hg=False, is_train=False),
                batch_size=self.cfg.get_batch_size(),
                shuffle=False,
                num_workers=self.cfg.get_num_workers(),
                pin_memory=True)
            print(">>> testing data loaded !")

    def validating(self):
        losses = utils.AverageMeter()

        self.model.eval()

        all_dist = []

        for i, (inps, tars) in enumerate(self.data_loader):
            inputs = Variable(inps.cuda())
            tars = Variable(tars.cuda(async=True))

            outputs = self.model(inputs)
            # calculate loss
            outputs_coord = outputs
            loss = self.loss_fuction(outputs_coord, tars)
            losses.update(loss.item(), inputs.size(0))
            # calculate erruracy
            targets_unnorm = self.unNormalizeData(tars.data.cpu().numpy(), self.stat_3d['mean'], self.stat_3d['std'],
                                                          self.stat_3d['dim_use'])
            outputs_unnorm = self.unNormalizeData(outputs.data.cpu().numpy(), self.stat_3d['mean'], self.stat_3d['std'],
                                                          self.stat_3d['dim_use'])

            # remove dim ignored
            dim_use = np.hstack((np.arange(3), self.stat_3d['dim_use']))

            outputs_use = outputs_unnorm[:, dim_use]
            targets_use = targets_unnorm[:, dim_use]

            #use procrustes analysis at testing
            if True:
                for ba in range(inps.size(0)):
                    gt = targets_use[ba].reshape(-1, 3)
                    out = outputs_use[ba].reshape(-1, 3)
                    _, Z, T, b, c = get_transformation(gt, out, True)
                    out = (b * out.dot(T)) + c
                    outputs_use[ba, :] = out.reshape(1, 51)

            sqerr = (outputs_use - targets_use) ** 2

            distance = np.zeros((sqerr.shape[0], 17))
            dist_idx = 0
            for k in np.arange(0, 17 * 3, 3):
                distance[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k + 3], axis=1))
                dist_idx += 1
            all_dist.append(distance)

        all_dist = np.vstack(all_dist)
        #joint_err = np.mean(all_dist, axis=0)
        ttl_err = np.mean(all_dist)
        print(">>> error: {} <<<".format(ttl_err))
        return losses.avg, ttl_err

    def tesing(self, action):
        print(">>> TEST on _{}_".format(action))
        self.data_loader = DataLoader(
            dataset=Human36M(actions=action, data_path=self.cfg.get_data_path(), use_hg=False, is_train=False),
            batch_size=self.cfg.get_batch_size(),
            shuffle=False,
            num_workers=self.cfg.get_num_workers(),
            pin_memory=True)
        _, err_test = self.validating()
        return err_test
    
    def unNormalizeData(self, normalized_data, data_mean, data_std, dimensions_to_use):
	    T = normalized_data.shape[0]  # Batch size
	    D = data_mean.shape[0]  # 96

	    orig_data = np.zeros((T, D), dtype=np.float32)

	    orig_data[:, dimensions_to_use] = normalized_data

	    # Multiply times stdev and add the mean
	    stdMat = data_std.reshape((1, D))
	    stdMat = np.repeat(stdMat, T, axis=0)
	    meanMat = data_mean.reshape((1, D))
	    meanMat = np.repeat(meanMat, T, axis=0)
	    orig_data = np.multiply(orig_data, stdMat) + meanMat
	    return orig_data