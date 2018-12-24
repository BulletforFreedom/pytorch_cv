#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:11:00 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""
import cv2

import torch
from torch.autograd import Variable

class Validating(object):
    def __init__(self, cfg, net, val_loader):
        self.cfg = cfg
        self.net = net
        self.val_loader=val_loader
        
    def __call__(self):
        size=0
        with torch.no_grad():
            correct = 0
            for (images, labels, image_dirs) in self.val_loader:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda(async=True))
                outputs = self.net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum()
                '''
                mask = torch.nonzero((predicted != labels))
                for i in mask:
                    img_dir=image_dirs[int(i)]
                    img=cv2.imread(img_dir)
                    img_name=img_dir.split('/')[-1]
                    cv2.imshow(img_name,img)
                    cv2.moveWindow(img_name,10,10)
                    k = cv2.waitKey(0)
                    if k==ord('q'):
                        cv2.destroyAllWindows()
                        break       
                    else:
                        cv2.destroyWindow(img_name)
                '''
                size+=predicted.size()[0]            
            acc = 100. * int(correct) / size
            print('测试分类准确率为：%.3f%%' %acc)            
            return acc