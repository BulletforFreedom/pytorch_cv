#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:17:04 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""

import cv2
import numpy as np

import global_matting as gm

class PostProgress(object):
    
    @staticmethod
    def mask_object(img):        
        #imgB=img[:,:,0]
        mask=[]
        mask_row=[]
        for i in img:
            for j in i:
                if j[1] == 5:
                    mask_row.append([0,0,0])
                else:
                    mask_row.append([255,255,255])
            mask.append(mask_row)
            mask_row=[]
        return mask
        mask=np.array(mask).astype('uint8')
        img_final=np.hstack((img,mask))
        name=img_dir.split('/')[-1]
        cv2.imshow(name,img_final)
        cv2.moveWindow(name,10,10)
        k = cv2.waitKey(0)
        if k==ord('q'):
            cv2.destroyAllWindows()
        else:
            cv2.destroyWindow(name)

    @staticmethod
    def smooth_edge(mask):
        smoothed_mask=[mask[0]]
        smoothed_mask_row=[mask[1][0]]
        for i in range(1, len(mask)-1):
            for j in range(1, len(mask[i])-1):
                if mask[i][j] != mask[i-1][j-1] or\
                mask[i][j] != mask[i][j-1] or\
                mask[i][j] != mask[i+1][j-1] or\
                mask[i][j] != mask[i-1][j] or\
                mask[i][j] != mask[i+1][j] or\
                mask[i][j] != mask[i-1][j+1] or\
                mask[i][j] != mask[i][j+1] or\
                mask[i][j] != mask[i+1][j+1]:
                    smoothed_mask_row.append([128, 128, 128])
                else:
                    smoothed_mask_row.append(mask[i][j])
            smoothed_mask_row.append(mask[i][-1])
            smoothed_mask.append(smoothed_mask_row)
            smoothed_mask_row=[mask[i+1][0]]
        smoothed_mask.append(mask[-1])
        smoothed_mask=np.array(smoothed_mask).astype('uint8')
        return smoothed_mask
        #img_final=np.hstack((img,mask))
        name='111'
        cv2.imshow(name,smoothed_mask)
        cv2.moveWindow(name,10,10)
        k = cv2.waitKey(0)
        if k==ord('q'):
            cv2.destroyAllWindows()
        else:
            cv2.destroyWindow(name)


if __name__=='__main__':
    img_seg_dir='/home/lsk/Downloads/pytorch_cv/data/Magic-wall/mask/ADE_val_00000035.png'
    img_seg=cv2.imread(img_seg_dir)
    img_path=img_seg_dir.split('.')[0]
    mask=PostProgress.mask_object(img_seg)
    smoothed_mask=PostProgress.smooth_edge(mask)
    alpha=gm.matting(img_seg, smoothed_mask[:,:,0])
    alpha=1-np.array(alpha)
    img_dir='/home/lsk/Downloads/pytorch_cv/data/Magic-wall/img/ADE_val_00000035.jpg'
    img=cv2.imread(img_dir)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0]=img_hsv[:, :, 0]*(1-alpha)+np.ones_like(img_hsv[:, :, 0])*(alpha)
    img_hsv[:, :, 1]=img_hsv[:, :, 1]*(1-alpha)+np.ones_like(img_hsv[:, :, 1])*(alpha)
    img_hsv[:, :, 2]=img_hsv[:, :, 2]*(1-0.5*alpha)+np.ones_like(img_hsv[:, :, 2])*(0.5*alpha)
    img_out=cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
    #gm.show(alpha)
    cv2.imwrite(img_path+'_output2.png',np.round(alpha*255).astype(np.uint8))
    cv2.imwrite(img_path+'_output.png',img_out)
    
    smoothed_mask1=1-smoothed_mask[:,:,0].astype(np.float)//255
    img_hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv1[:, :, 0]=img_hsv1[:, :, 0]*(1-smoothed_mask1)+np.ones_like(img_hsv1[:, :, 0])*(smoothed_mask1)
    img_hsv1[:, :, 1]=img_hsv1[:, :, 1]*(1-smoothed_mask1)+np.ones_like(img_hsv1[:, :, 1])*(smoothed_mask1)
    img_hsv1[:, :, 2]=img_hsv1[:, :, 2]*(1-0.5*smoothed_mask1)+np.ones_like(img_hsv1[:, :, 2])*(0.5*smoothed_mask1)
    img_out1=cv2.cvtColor(np.round(img_hsv1).astype(np.uint8), cv2.COLOR_HSV2BGR)
    #gm.show(alpha)
    cv2.imwrite(img_path+'_output1.png',img_out1)
    
    img_mask_dir1='/home/lsk/Downloads/pytorch_cv/data/Magic-wall/result/ADE_val_00000035.png'
    alpha1=cv2.imread(img_mask_dir1)
    img_mask_dir2='/home/lsk/Downloads/pytorch_cv/data/Magic-wall/mask/ADE_val_00000035_output2.png'
    alpha2=cv2.imread(img_mask_dir2)
