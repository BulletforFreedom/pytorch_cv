#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:27:22 2018

@author: lsk
"""


import cv2
import numpy as np
import time

import torch as t
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

from common.logger import Logger as log
from common.util import Util
from cfg.configer import Configer as Configer_y
from cfg.configer import Configer as Configer_r
from classification.testing import Testing
from detection import detector
from net.darknet import Darknet

class person(object):
    def __init__(self, center, height, width, duration, around_env):
        self.center=center
        self.height=height
        self.width=width
        self.duration=duration
        self.around_env=around_env   

if __name__=='__main__':
    
    #yolo
    y_cfg=Configer_y("yolov3.cfg")
    net_info=y_cfg.get_net_info()
    model = Darknet(y_cfg)    
    model.load_weights("/home/lsk/Downloads/pytorch_cv/project/backup/yolov3.weights")
    log.info('yolov3.weights Done!')    
    model = nn.DataParallel(model)
    model.cuda()    
    model.eval()    
    DK=detector.DK_Output()
    
    #resnet
    r_cfg = Configer_r("resnet34.cfg")
    net = models.resnet34(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, r_cfg.get_num_classes())
    ckpt = t.load("/home/lsk/Downloads/pytorch_cv/project/backup/education_resnet34_2018_12_19_16_37.pth.tar")
    net.load_state_dict(ckpt['state_dict'])
    log.info('resnet34 ckpt Done!')
    net = net.cuda()
    net.eval()
    tester=Testing(r_cfg, net)
    
    #video
    video_path='/home/lsk/Downloads/pytorch_cv/data/education/videos/1.MOV'    
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')    
    rate = cap.get(1)
    ret, frame = cap.read()
    #size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    h,w,c = frame.shape
    out = cv2.VideoWriter('/home/lsk/Downloads/pytorch_cv/data/education/videos/output.avi',
                          fourcc, 15, (w, h))
    stored_frames=[]
    
    while ret:
        frame_start=time.time()
        img = cv2.resize(frame, (416, 416)) 
        img =  img[:,:,::-1].transpose((2,0,1))
        img = img[np.newaxis,:,:,:]/255.0
        img = t.from_numpy(img).float()
        img = Variable(img.cuda())
        
        detection_start=time.time()
        #object detection
        prediction = model(img)
        results=DK.write_results(prediction, y_cfg.get_num_classes())
        results=results.cpu().detach().numpy()
        detection_end=time.time()-detection_start
        print('detection_time: {detection_time:.4}ms'
              .format(detection_time=detection_end*1000))
        
        im_gt = frame.copy()
        
        frame_info=[]
        #crop objects
        sum_start=time.time()
        for x in results:
            if int(x[-1])!=0:
                continue
            classes=[]
            #extra roi
            gt_boxe=[x[1], x[2], x[3], x[4]]
            delta_x=(gt_boxe[2]-gt_boxe[0])*0.1
            delta_y=(gt_boxe[3]-gt_boxe[1])*0.1
            x1 = int(max([0, ((x[1]-delta_x)/416)*w ]))
            x2 = int(min([w, ((x[3]+delta_x)/416)*w ]))
            y1 = int(max([0, ((x[2]-delta_y)/416)*h ]))
            y2 = int(min([h, ((x[4]+delta_y)/416)*h ]))
            roi = im_gt[y1:y2, x1:x2]
            
            #image classification
            classify_start=time.time()
            pred, confidence=tester(roi)
            classify_end=time.time()-classify_start
            print('classify_time: {classify_time:.4}ms'
              .format(classify_time=classify_end*1000))
            
            Post_process_start=time.time()
            classes.append(pred)
            for stored_frame in stored_frames:
                for info in stored_frame:
                    iou=Util.calculate_iou(info[0:4], [x1, y1, x2, y2])
                    if iou>0.8:
                        classes.append(info[4])
                        confidence+=info[5]
            confidence /= len(classes)
            frame_info.append([x1, y1, x2, y2, pred, confidence])#!!!dou dong wei jie jue
            
            if confidence<0.75:
                gt_class_name='None'
            else:
                pred=max(classes, key=lambda x: classes.count(x))#!!!chu xian ci shu xiang tong de wen ti wei jie jue
                gt_class_name=r_cfg.get_dataset_name_seq()[pred]
                
            cv2.rectangle(im_gt,(x1,y1),(x2,y2),(0,0,255),1)
            cv2.putText(im_gt,
                    '%s_%.4f'%(gt_class_name, confidence),
                    (x1,int(y1*1.1)),
                    cv2.FONT_HERSHEY_SIMPLEX,1,
                    (0,200,255),
                    thickness=2)
            Post_process_end=time.time()-Post_process_start
            print('Post_process_time: {Post_process_time:.4}ms'
              .format(Post_process_time=Post_process_end*1000))
        sum_end=time.time()-sum_start
        print('sum_time: {sum_time:.4}ms'
              .format(sum_time=sum_end*1000))
        plot_start=time.time()
        if len(stored_frames)==20:
            del stored_frames[0]
        stored_frames.append(frame_info)
        win_name = video_path
        cv2.imshow(win_name,im_gt)
        out.write(im_gt)
        ret, frame = cap.read()
        frame_end=time.time()-plot_start
        print('plot_time: {plot_time:.4}ms'
              .format(plot_time=frame_end*1000))
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_end=time.time()-frame_start
        print('frame_time: {frame_time:.4}ms'
              .format(frame_time=frame_end*1000))
        #break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    t.cuda.empty_cache()
    
'''
def foo(*args, **kwargs):
     print('args =', args)
     print('kwargs = ', kwargs)
     print('-----------------------')
class A():
    def __init__(self):
        print('__init__')
        
    def __getitem__(self, index):
        print(index)
        
    #def __del__(self):
        #print('__del__')

a=A()
if __name__ == '__main__':
    foo(1, 2, 3, 4)
    foo(a=1, b=2, c=3)
    foo(1,2,3,4, a=1, b=2, c=3)
    foo('a', 1, None, a=1, b='2', c=3)
'''