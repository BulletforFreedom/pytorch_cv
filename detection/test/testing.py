#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 18:10:46 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""

import os,sys

import cv2
import json

import torch as t
import torch.nn as nn

from logger import Logger as log
import util as ut
import detector
from cfg.tools.configer import Configer
from data_loader.det_data_loader import DetDataLoader as DataLoader
from net.darknet import Darknet

class Testing(object):
    def __init__(self, saved_weights):
        self.cfg = Configer("../cfg/yolov3.cfg")
        weightfile = os.path.join(self.cfg.get_backup_path(),saved_weights+'.pkl')
        self.model = Darknet(self.cfg)
        log.info('Loading weights from %s...'%weightfile)
        state_dict = t.load(weightfile)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
            
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params    
        self.model.load_state_dict(new_state_dict,strict = False)
        log.info('Done!')
        
        self.model = nn.DataParallel(self.model)
        if t.cuda.is_available():        
            self.model.cuda()         
            
        self.model.eval()
        
        self.DK=detector.DK_Output()
        
    def plot_predicted_img(self):
        dataloader = DataLoader(self.cfg)
        self.test_loader = dataloader.get_loader() 
        # Start the training loop   
        for step, (images, img_dir_list, gt_bboxes, gt_labels) in enumerate(self.test_loader):

            images=images.cuda()            
            prediction = self.model(images)
            #origin_results=loss_function.debug_loss(prediction, gt_labels, gt_bboxes)            
            results=self.DK.write_results(prediction, self.cfg.get_num_classes())
            results=results.cpu().detach().numpy()
            t.cuda.empty_cache()
            
            for i, img_dir in enumerate(img_dir_list):
                im = cv2.imread(img_dir)
                im_gt = im.copy()            
                gt_boxes = [[x[1], x[2], x[3], x[4]] for x in results if x[0]==i]
                gt_class_ids=[int(x[-1]) for x in results if x[0]==i]
                gt_class_name=[ self.cfg.get_dataset_name_seq()[x] for x in gt_class_ids]
                im = ut.plot_bb(im_gt,gt_boxes,gt_class_name,self.cfg.get_final_inp_dim())
                win_name = img_dir
                cv2.imshow(win_name,im)
                cv2.moveWindow(win_name,10,10)
                
                k = cv2.waitKey(0)
                if k==ord('q'):
                    cv2.destroyAllWindows()
                    break
                elif k==ord('c'):
                    try:
                        cv2.destroyWindow(win_name)
                    except:
                        cv2.destroyAllWindows()
                        break
            if step == 1:
                break
            
    def prediction_result(self):
        dataloader = DataLoader(self.cfg, True)
        self.test_loader = dataloader.get_loader() 
        for step, (images, img_dir_list, gt_bboxes, gt_labels) in enumerate(self.test_loader):

            images=images.cuda()            
            prediction = self.model(images)       
            results=self.DK.write_results(prediction, self.cfg.get_num_classes())
            results=results.cpu().detach().numpy()
            for i, x in enumerate(img_dir_list):
                json_dict = dict()
                file_name = x.split('/')[-1]
                json_dict['width'] = self.coco.imgs[img_id]['width']
                json_dict['height'] = self.coco.imgs[img_id]['height']
    
                ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
                annos = self.coco.loadAnns(ann_ids)
                object_list = list()
                for anno in annos:
                    object_dict = dict()
                    object_dict['label'] = self.cat_ids.index(anno['category_id'])
                    bbox = anno['bbox']
                    gt_x = (float(bbox[0]) + float(bbox[2])/2)/self.coco.imgs[img_id]['width']
                    gt_y = (float(bbox[1]) + float(bbox[3])/2)/self.coco.imgs[img_id]['height']
                    gt_w = float(bbox[2])/self.coco.imgs[img_id]['width']
                    gt_h = float(bbox[3])/self.coco.imgs[img_id]['height']
                    object_dict['bbox'] = [gt_x, gt_y, gt_w, gt_h]
    
                    object_list.append(object_dict)
    
                json_dict['objects'] = object_list
                fw = open(os.path.join(self.json_dir, '{}.json'.format(file_name.split('.')[0])), 'w')
                fw.write(json.dumps(json_dict))
                fw.close()
                shutil.copy(os.path.join(self.args.ori_img_dir, file_name),
                            os.path.join(self.image_dir, file_name))
        t.cuda.empty_cache()
        
    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        self.transpose = self.categories#self.data_coco['categories']
        # print(self.transpose)
        print(self.map_dic)
        # 保存json文件
        if self.is_gt == False:
            for item in self.data_coco['annotations']:
                item['category_id'] = self.transpose[item['category_id'] -
                                                     1]['supercategory']
                for i in self.map_dic:
                    if i['supercategory'] == item['category_id']:
                        item['category_id'] = i['id']
                        break
                        # print(item['category_id'])
                        # print(i['id'])
                # print(item)
        json.dump(self.data_coco, open(self.save_json_path, 'w'),
                  indent=4)  # indent=4 更加美观显示

if __name__=='__main__':
    weights_name='98000_params'
    tester=Testing(weights_name)
