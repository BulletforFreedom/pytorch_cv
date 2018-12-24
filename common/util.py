"""
Created on Tue Aug 14 18:10:46 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""

import os,sys
import numpy as np
import cv2 
import random
import glob
import time

import torch 
from torch.autograd import Variable

sys.path.append('/home/lsk/Downloads/pytorch_cv/project')
from common.logger import Logger as log

class Util(object):
    
    @staticmethod
    def calculate_iou(box1, box2):
        '''
        cpu
        '''
        inter_rect_x1 =  max(box1[0], box2[0])
        inter_rect_y1 =  max(box1[1], box2[1])
        inter_rect_x2 =  min(box1[2], box2[2])
        inter_rect_y2 =  min(box1[3], box2[3])    
        # Intersection area
        inter_area =    max(inter_rect_x2 - inter_rect_x1 + 1, 0) * \
                        max(inter_rect_y2 - inter_rect_y1 + 1, 0)
        # Union Area
        b1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        b2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou
    
    @staticmethod
    def unique(tensor1d):
         temp=[]
         for x in tensor1d:
             if len(temp)==0:
                 temp.append(x)
             else:
                 for y in temp:
                     if x==y:
                         break
                     if y==temp[-1]:
                         temp.append(x)
         return temp
    
    @staticmethod
    def load_classes(namesfile):
        fp = open(namesfile, "r")
        names = fp.read().split("\n")[:-1]
        return names
    
    @staticmethod
    def get_test_input(img_name,inp_dim):
        img = cv2.imread(img_name)
        img = cv2.resize(img, (inp_dim,inp_dim)) #Resize to the input dimension
        img_ = img[:,:,::-1].transpose((2,0,1)) # BGR -> RGB | H X W X C -> C X H X W 
        img_ = img_[np.newaxis,:,:,:]/255.0 #Add a channel at 0 (for batch) | Normalise
        img_ = torch.from_numpy(img_).float() #Convert to float
        img_ = Variable(img_) # Convert to Variable
        return img_
    
    @staticmethod
    def plot_bb(im,bb,cls,inp_dim,textSize=1,textThickness=2):
        if im.shape[0]==3:
            im=im.transpose(1,2,0)
        h,w,c = im.shape
        for idx,box in enumerate(bb):
            
            x1 = int(max([0, (box[0]/inp_dim)*w ]))
            x2 = int(min([w, (box[2]/inp_dim)*w ]))
            y1 = int(max([0, (box[1]/inp_dim)*h ]))
            y2 = int(min([h, (box[3]/inp_dim)*h ]))
    
            cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),1)
            cv2.putText(im,
                        '%s'%(cls[idx]),
                        (x1,int(y1*1.1)),
                        cv2.FONT_HERSHEY_SIMPLEX,textSize,
                        (0,200,255),
                        thickness=textThickness) 
        return im
    
    @staticmethod
    def random_resize_image(original_img_size, strides):
        amplitude = original_img_size // 32 -5         
        new_size = (random.randint(0,9) + amplitude) * strides
        return new_size
    
    @staticmethod
    # ################################
    # Format Change
    # ################################
    def convert_bb_format(boxes):
        '''from (cx,cy,w,h)-->(x1,y1,x2,y2)
    
        :param boxes:
        :return:
        '''
        new_boxes = []
    
        for box in boxes:
            b_w = box[2]
            b_h = box[3]
            c_x = box[0]
            c_y = box[1]
    
            x1 = (max([0., (c_x - 0.5 * b_w)]))
            x2 = (min([1., (c_x + 0.5 * b_w)]))
            y1 = (max([0., (c_y - 0.5 * b_h)]))
            y2 = (min([1., (c_y + 0.5 * b_h)]))
    
            new_boxes.append(np.array([x1,y1,x2,y2]))
    
        new_boxes = np.array(new_boxes)
    
        return new_boxes
    
    @staticmethod
    def get_file_name_in_dir(in_dir,out_path,suffix='*.jpg'):
        '''
        Get the file names in "in_dir"
    
        :param in_dir: file dir
        :param out_path: save path
        :param suffix: file suffix
        :return: None
        '''
        filelist = glob.glob(os.path.join(in_dir, suffix))
        f = open(out_path,'w')
        for line in filelist:
            name = line.split('/')[-1].split('.')[0]
            f.write(name+'\n')
        f.close()
    
        return 0
    
    @staticmethod
    def get_file_full_path_in_dir(files_dir,out_path,suffix='*.jpg'):
        '''
        Get file full path from directory "files_dir"
    
        :param files_dir:
        :param out_path:
        :param suffix:
        :return:
        '''
        filelist = glob.glob(os.path.join(files_dir, suffix))
        f = open(out_path, 'w')
        for line in filelist:
            f.write(line + '\n')
        f.close()
    
    @staticmethod
    def balance_lightness(bgr,gridsize=8): #bgr = cv2.imread(in_image_path)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        
        lab_planes = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
        
        lab_planes[0] = clahe.apply(lab_planes[0])
        
        lab = cv2.merge(lab_planes)
        
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return bgr
    
    @staticmethod
    def save_ckpt(cfg, state):
        #ckpt_best_method_year_month_day_time
        ckpt_name=cfg.get_project_name()+'_'+cfg.get_method()+'_'+cfg.get_start_time()+'.pth.tar'
        file_dir = os.path.join(cfg.get_backup_path(), ckpt_name)
        torch.save(state, file_dir)
        log.info('Saved at %s' %file_dir)
    
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0     
        self.max_arrow=50

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar) #这两句打印字符到终端
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('\nDone')
        self.i = 0
    
class ProgressBar:
    def __init__(self, count = 0, total = 0, width = 50):
        self.count = count
        self.total = total
        self.width = width
    def move(self):
        self.count += 1
    def log(self, s):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        print(s)
        progress = self.width * self.count // self.total
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('#' * progress + '-' * (self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()
    
if __name__ == '__main__':
    bar = ProgressBar(total = 10)
    for i in range(10):
        bar.move()
        bar.log('We have arrived at: ' + str(i + 1))
        time.sleep(1)
