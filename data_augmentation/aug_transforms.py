#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:21:57 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import cv2
import numpy as np
from PIL import Image, ImageOps

#from common.logger import Logger as Log
from project.cfg.configer import Configer


class RandomPad(object):
    """ Padding the Image to proper size.
            Args:
                stride: the stride of the network.
                pad_value: the value that pad to the image border.
                img: Image object as input.

            Returns:
                img: Image object.
    """
    def __init__(self, pad_border=None, pad_ratio=0.5):
        assert isinstance(pad_border, int)
        self.pad_border = pad_border
        self.ratio = pad_ratio

    def __call__(self, img, label=None, mask=None, kpts=None, bboxes=None):
        assert isinstance(img, Image.Image)
        assert label is None or isinstance(label, Image.Image)
        assert mask is None or isinstance(mask, Image.Image)

        rand_value = random.randint(1, 100)
        if rand_value > 100 * self.ratio:
            return img, label, mask, kpts, bboxes

        left_pad = random.randint(-self.pad_border, self.pad_border)  # pad_left
        up_pad = random.randint(-self.pad_border, self.pad_border)  # pad_up
        right_pad = -left_pad  # pad_right
        down_pad = -up_pad  # pad_down

        img = ImageOps.expand(img, (left_pad, up_pad, right_pad, down_pad), fill=(128, 128, 128))

        if label is not None:
            label = ImageOps.expand(label, (left_pad, up_pad, right_pad, down_pad), fill=255)

        if mask is not None:
            mask = ImageOps.expand(mask, (left_pad, up_pad, right_pad, down_pad), fill=1)

        if kpts is not None and len(kpts) > 0:
            num_objects = len(kpts)
            num_keypoints = len(kpts[0])

            for i in range(num_objects):
                for j in range(num_keypoints):
                    kpts[i][j][0] += left_pad
                    kpts[i][j][1] += up_pad

        if bboxes is not None and len(bboxes) > 0:
            for i in range(len(bboxes)):
                bboxes[i][0] += left_pad
                bboxes[i][1] += up_pad
                bboxes[i][2] += left_pad
                bboxes[i][3] += up_pad

        return img, label, mask, kpts, bboxes

class RandomCrop(object):
    '''
    随机裁剪
    area_ratio为裁剪画面占原画面的比例
    hw_vari是扰动占原高宽比的比例范围
    '''
    def __init__(self, area_ratio=0.8, hw_vari=0.1):
        self.area_ratio = area_ratio
        self.hw_vari = hw_vari
        self.crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]
    def __call__(self, img):
        
        h, w = img.shape[:2]
        hw_delta = np.random.uniform(-self.hw_vari, self.hw_vari)
        hw_mult = 1 + hw_delta
    	
    	# 下标进行裁剪，宽高必须是正整数
        w_crop = int(round(w*np.sqrt(self.area_ratio*hw_mult)))
    	
    	# 裁剪宽度不可超过原图可裁剪宽度
        if w_crop > w:
            w_crop = w
    		
        h_crop = int(round(h*np.sqrt(self.area_ratio/hw_mult)))
        if h_crop > h:
            h_crop = h
    	
    	# 随机生成左上角的位置
        x0 = np.random.randint(0, w-w_crop+1)
        y0 = np.random.randint(0, h-h_crop+1)
    	
        return self.crop_image(img, x0, y0, w_crop, h_crop)

class RandomRotate(object):
    '''
    随机旋转
    angle_vari是旋转角度的范围[-angle_vari, angle_vari)
    p_crop是要进行去黑边裁剪的比例
    '''
    def __init__(self, angle_vari, p_crop):
        self.angle_vari=angle_vari
        self.p_crop=p_crop
        self.crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]
        
    def __call__(self, img):
        angle = np.random.uniform(-self.angle_vari, self.angle_vari)
        crop = False if np.random.random() > self.p_crop else True
        return self._rotate_image(img, angle, crop)
    
    def _rotate_image(self, img, angle, crop):
        '''
        定义旋转函数：
        angle是逆时针旋转的角度
        crop是个布尔值，表明是否要裁剪去除黑边
        '''
        h, w = img.shape[:2]
    	
    	# 旋转角度的周期是360°
        angle %= 360
    	
    	# 用OpenCV内置函数计算仿射矩阵
        M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    	
    	# 得到旋转后的图像
        img_rotated = cv2.warpAffine(img, M_rotate, (w, h))
    
    	# 如果需要裁剪去除黑边
        if crop:
            angle_crop = angle % 180             	    # 对于裁剪角度的等效周期是180°
            if angle_crop > 90:                        	# 并且关于90°对称
                angle_crop = 180 - angle_crop
    		
            theta = angle_crop * np.pi / 180.0    		# 转化角度为弧度
            hw_ratio = float(h) / float(w)    		    # 计算高宽比
    		
    
            tan_theta = np.tan(theta)                   # 计算裁剪边长系数的分子项
            numerator = np.cos(theta) + np.sin(theta) * tan_theta
    		
    
            r = hw_ratio if h > w else 1 / hw_ratio		# 计算分母项中和宽高比相关的项
            denominator = r * tan_theta + 1		 		# 计算分母项
    
            crop_mult = numerator / denominator			# 计算最终的边长系数
            w_crop = int(round(crop_mult*w))			# 得到裁剪区域
            h_crop = int(round(crop_mult*h))
            x0 = int((w-w_crop)/2)
            y0 = int((h-h_crop)/2)
            img_rotated = self.crop_image(img_rotated, x0, y0, w_crop, h_crop)
        return img_rotated
    
class RandomHSV(object):
    '''
    随机hsv变换
    hue_vari是色调变化比例的范围
    sat_vari是饱和度变化比例的范围
    val_vari是明度变化比例的范围
    '''
    def __init__(self, hue_vari=10, sat_vari=0.1, val_vari=0.1):
        self.hue_vari=hue_vari
        self.sat_vari=sat_vari
        self.val_vari=val_vari
        
    def __call__(self, img):         
        hue_delta = np.random.randint(-self.hue_vari, self.hue_vari)
        sat_mult = 1 + np.random.uniform(-self.sat_vari, self.sat_vari)
        val_mult = 1 + np.random.uniform(-self.val_vari, self.val_vari)
        return self._hsv_transform(img, hue_delta, sat_mult, val_mult)

    def _hsv_transform(self,img, hue_delta, sat_mult, val_mult):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180        #Hue色调
        img_hsv[:, :, 1] *= sat_mult                                   #Saturation饱和度
        img_hsv[:, :, 2] *= val_mult                                   #Value明度亮度
        img_hsv[img_hsv > 255] = 255
        return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)

class RandomFlip(object):
    """Flip an Image
    Args:
        img (Image): Image.
        orientation(orientation): Horizontal: 1, Vertical:0, Horizontal & Vertical: -1

    Returns:
        img: flipped image.
    """
    def __init__(self, orientation=1):
        self.orientation=orientation
    def __call__(self, img):
        if random.randint(0, 1):
            img=cv2.flip(img, self.orientation)
        return img

class RandomGamma(object):
    '''
    随机gamma变换
    gamma_vari是Gamma变化的范围[1/gamma_vari, gamma_vari)
    '''
    def __init__(self, gamma_vari):
        self.gamma_vari=gamma_vari
        
    def __call__(self, img):
        log_gamma_vari = np.log(self.gamma_vari)
        alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
        gamma = np.exp(alpha)
        return self._gamma_transform(img, gamma)
    
    def _gamma_transform(img, gamma):
        '''
        定义gamma变换函数：
        gamma就是Gamma
        '''
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)
    
class AugCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> CV2AugCompose([
        >>>     RandomCrop(),
        >>> ])
    """
    
    def __init__(self, configer):
        self.augmentations=[]
        augs = configer.get_augmentation()
        if 'randomHSV' in augs:
            self.augmentations.append(RandomHSV())
        if 'randomflip' in augs:
            self.augmentations.append(RandomFlip())
        if 'randomcrop' in augs:
            self.augmentations.append(RandomCrop())
            
    def __call__(self, img):
        for aug in self.augmentations:
            img = aug(img)
        return img
        
if __name__=='__main__':
    cfg=Configer("/home/lsk/Downloads/pytorch_cv/project/cfg/resnet34.cfg")
    ac=AugCompose(cfg)
    name='112.jpg'
    img=cv2.imread('/home/lsk/Downloads/pytorch_cv/data/education/hands_up/112.jpg')
    img_flip=ac(img)
    # --plot
    im_final = cv2.resize(img_flip, 
                         (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    im_final = np.hstack((img,im_final))
    im_final = cv2.resize(im_final, 
                          (im_final.shape[1] // 2, im_final.shape[0] // 2),
                          interpolation=cv2.INTER_LINEAR)
    cv2.imshow(name,im_final)
    #cv2.moveWindow(name,10,10)
    k = cv2.waitKey(0)
    if k==ord('q'):
        cv2.destroyAllWindows()
        #break            
    elif k==ord('v'):
        #shutil.copyfile(orig_path, os.path.join(dest_dir, name + '.jpg'))
        cv2.destroyWindow(name)
    else:
        cv2.destroyWindow(name)