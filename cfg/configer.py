# -*- coding: utf-8 -*-
import os,sys
import glob
import time

sys.path.append('/home/lsk/Downloads/pytorch_cv/project')
from data.data_info import education as edu_data_info
from data.data_info import coco2014 as coco14_data_info
from common.logger import Logger as log

class Configer(object):
    def __init__(self, cfgfile):
        self.cfgfile=os.path.join('/home/lsk/Downloads/pytorch_cv/project/cfg', cfgfile)
        self.blocks = []#collections.OrderedDict()
        self.net_info=None
        self.aug=None
        self._parse_cfg()
        self.net_info['start_time']=time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time()))
        self.net_info['dataset']=self._set_data_info()
        self.net_info['strides'] = -1
        self.net_info['scaled_anchor_list'] = []
        self.net_info['anchor_list'] = []
        self.net_info['num_feature_map'] = 0
        self.net_info['resize_dim'] = int(self.net_info['inp_dim'])
        self.net_info['current_path']=os.path.abspath('.')
        
    def set_resize_dim(self, new_size):
        self.net_info['resize_dim'] = new_size        
        
    def count_num_feature_map(self):
        self.net_info['num_feature_map'] += 1
        
    def set_total_strides(self, strides):
        self.net_info['strides'] = strides
        
    def set_scaled_anchor_list(self, scaled_anchor_list):
        self.net_info['scaled_anchor_list'].append(scaled_anchor_list)    
        
    def set_anchor_list(self, anchor):
        self.net_info['anchor_list'].append(anchor)
        
    def _set_data_info(self):
        if self.get_project_name()=='coco2014':
            return coco14_data_info
        elif self.get_project_name()=='education':
            return edu_data_info
        else:
            log.error("Project name is not found")
            sys.exit()
    
    def get_start_time(self):
        return self.net_info['start_time']
    
    def get_resize_dim(self):
        return self.net_info['resize_dim']
    
    def get_optimizer(self):
        return self.net_info['optimizer']
    
    def get_lr_steps(self):
        return int(self.net_info['epochs'])//4
    
    def get_lr_decay(self):
        return float(self.net_info['lr_decay'])
    
    def get_lr_gamma(self):
        return float(self.net_info['lr_gamma'])
        
    def get_itr(self):
        return self.net_info['next_itr']
    
    def get_num_feature_map(self):
        return self.net_info['num_feature_map']
    
    def get_net_info(self):
        return self.net_info
    
    def get_project_name(self):
        return self.net_info['project_name']
    
    def is_train(self):
        return int(self.net_info['train'])
    
    def is_mul_train(self):
        return int(self.net_info['mul_train'])
    
    def get_blocks(self):
        return self.blocks#['net']
    
    def get_final_inp_dim(self):
        return int(self.net_info['inp_dim'])

    def get_out_dim(self):
        return int(self.net_info['out_dim'])

    def get_num_classes(self):
        return len(self.net_info['dataset']['name_seq'])
    
    def get_loss_lambda(self):
        return [float(self.net_info['coord_scale']), float(self.net_info['object_scale']),
                float(self.net_info['noobject_scale']), float(self.net_info['class_scale'])]
    
    def get_iou_threshold(self):
        return float(self.net_info['ignore_iou_thresh'])
    
    def get_total_strides(self):
        return self.net_info['strides']
    
    def get_anchors(self):
        return self.net_info['anchors']
    
    def get_anchor_list(self):
        return self.net_info['anchor_list']
        
    def get_scaled_anchor_list(self):
        return self.net_info['scaled_anchor_list']
    
    def get_method(self):
        return self.net_info['method']
    
    def get_num_workers(self):
        return int(self.net_info['workers'])
    
    def get_batch_size(self):
        return int(self.net_info['batch_size'])
    
    def get_epochs(self):
        return int(self.net_info['epochs'])

    def get_learning_rate(self):
        return float(self.net_info['learning_rate'])
    
    def get_weight_decay(self):
        return float(self.net_info['weight_decay'])
    
    def get_dataset_mean(self):
        return self.net_info['dataset']['mean']
    
    def get_dataset_std(self):
        return self.net_info['dataset']['std']
    
    def get_dataset_name_seq(self):
        return self.net_info['dataset']['name_seq']
    
    def get_data_path(self):
        return os.path.join(self.net_info['current_path'],
                            'data',
                            self.get_project_name(),
                            'train_set')
    
    def get_current_path(self):
        return self.net_info['current_path']
    
    def get_backup_path(self):
        backup_path = os.path.join(self.net_info['current_path'],'backup')
        if not os.path.exists(backup_path):
            os.mkdir(backup_path)
        return backup_path

    def get_ckpt(self):
        ckpt_dir=glob.glob(os.path.join(self.get_backup_path(),
                              self.net_info['ckpt']+'.*'))
        if not len(ckpt_dir):
            log.error("{} is not found".format(ckpt_dir))
            sys.exit()
        if len(ckpt_dir)!=1:
            log.error("File {} conflict!".format(self.net_info['ckpt']))
            sys.exit()
        ckpt_dir=ckpt_dir[0]        
        log.info(">>> loading ckpt from '{}'".format(ckpt_dir))
        return ckpt_dir

    def get_augmentation(self):
        return self.aug
        
    def _parse_cfg(self):
        """
        Takes a configuration file
        
        Returns a list of blocks. Each blocks describes a block in the neural
        network to be built. Block is represented as a dictionary in the list
        
        """
        file = open(self.cfgfile, 'r')
        lines = file.read().split('\n')     #store the lines in a list
        lines = [x for x in lines if len(x) > 0] #get read of the empty lines 
        lines = [x for x in lines if x[0] != '#']  
        lines = [x.strip() for x in lines]
    
        
        block = {}
        
        for line in lines:
            if line[0] == "[":               #This marks the start of a new block
                if len(block) != 0:
                    self.blocks.append(block)
                    block = {}
                block["type"] = line[1:-1].rstrip()                
            else:
                key,value = line.split("=")
                    
                block[key.rstrip()] = value.lstrip()
        self.blocks.append(block)
        self.net_info=self.blocks.pop(0)
        self.aug=self.blocks.pop(0)
        '''
        block_name = None
        
        
        for line in lines:
            if line[0] == "[":               #This marks the start of a new block
                if len(block) != 0:
                    self.blocks[block_name]=block
                    block = {}
                block_name = line[1:-1].rstrip()
            else:
                key,value = line.split("=")
                block[key.rstrip()] = value.lstrip()
        self.blocks[block_name]=block
        self.net_info=self.blocks.pop('Hyperparametric')
        '''

if __name__ == '__main__':
    cfg=Configer("/home/lsk/Downloads/pytorch_cv/project/cfg/resnet34.cfg")
    net_info=cfg.get_net_info()