#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:47:44 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""
import cv2
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Testing(object):
    def __init__(self, cfg, net):
        self.cfg = cfg
        self.net = net
               
        
        
    def __call__(self, image):
        with torch.no_grad():
            time1=time.time()
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image[:,:,:]/255.0
            image = cv2.resize(image, (224,224), interpolation=cv2.INTER_LINEAR)
            image=torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()                
            image = Variable(image.cuda())
            time2=time.time()
            outputs = self.net(image)
            time3=time.time()
            # 取得分最高的那个类 (outputs.data的索引号)
            confidence=F.softmax(outputs,dim=1)
            confidence, predicted = torch.max(confidence.data, 1)
            print('1time: {t1:.4}ms'
              .format(t1=(time2-time1)*1000))
            print('net_time: {t1:.4}ms'
              .format(t1=(time3-time2)*1000))
            return predicted, confidence
            #ffmpeg -i 1.mp4 -r 30  image-%05d.jpeg        