# -*- coding: utf-8 -*-
import numpy as np

import torch as t
import torch.nn as nn
import torch.nn.functional as F

#from common.logger import Logger as log
#from common import util as ut

class EmptyLayer(nn.Module):
    def __init__(self,shortcut_from=-1,route_start=-1,route_end=-1,anchors=[]):
        super(EmptyLayer,self).__init__()
        self.shortcut_from=shortcut_from
        self.route_start=route_start
        self.route_end=route_end
        self.anchors=anchors
    
    def get_shortcut_from(self):
        return self.shortcut_from
    
    def get_route_parm(self):
        return self.route_start,self.route_end
    
    def get_anchors(self):
        return self.anchors
     
class UpsamplingLayer(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(UpsamplingLayer, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
    
class DetectionLayer(nn.Module):
    def __init__(self, anchors, configer, CUDA=True):
        super(DetectionLayer,self).__init__()
        self.anchors=anchors
        self.configer=configer
        self.CUDA=CUDA
        self.stride=-1
        self.scaled_anchors=[]
    
    def forward(self, x):
        #x = x.data
        #global CUDA
        prediction = self._predict_transform(x)
        return prediction

    def _predict_transform(self,prediction):
        #inp_dim: weith or height of input image
        #stride: Scaling ratio        
        batch_size = prediction.size(0)       
        if self.stride == -1:
            self.stride=self.configer.get_resize_dim()//prediction.size(2)
        if self.configer.get_total_strides()==-1:            
            self.configer.set_total_strides(self.stride)
        grid_size=prediction.size(2)
        bbox_attrs = 5 + self.configer.get_num_classes()
        num_anchors = len(self.anchors)
        if len(self.configer.get_scaled_anchor_list())<self.configer.get_num_feature_map(): #len(self.scaled_anchors)==0 or             
            self.scaled_anchors = [(a[0]/self.stride, a[1]/self.stride) for a in self.anchors]            
            self.configer.set_scaled_anchor_list(self.scaled_anchors)
            #log space transform height and the width
            self.scaled_anchors=t.FloatTensor(self.scaled_anchors)
            if self.CUDA:
                self.scaled_anchors=self.scaled_anchors.cuda()
            self.scaled_anchors=self.scaled_anchors.repeat(grid_size*grid_size,1).unsqueeze(0)
            
        prediction = prediction.view(batch_size,bbox_attrs*num_anchors,grid_size*grid_size)
        prediction = prediction.transpose(1,2).contiguous()
        prediction = prediction.view(batch_size,grid_size*grid_size*num_anchors,bbox_attrs)
        
        #Sigmoid the centre_X, centre_Y
        prediction[:,:,:2] = t.sigmoid(prediction[:,:,:2])
        #Add the center offsets
        
        if not self.configer.is_train():
            grid_len = np.arange(grid_size)
            a,b=np.meshgrid(grid_len,grid_len)
            x_offset=t.FloatTensor(a).view(-1,1)
            y_offset=t.FloatTensor(b).view(-1,1)
            if self.CUDA:
                x_offset=x_offset.cuda()
                y_offset=y_offset.cuda()
            x_y_offset=t.cat((x_offset,y_offset),1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
            prediction[:,:,:2] += x_y_offset           
            
            prediction[:,:,2:4]=t.exp(prediction[:,:,2:4])*self.scaled_anchors 
            
            prediction[:,:,:4] *= self.stride            
            
        #Sigmoid the object confidencce
        prediction[:,:,4] = t.sigmoid(prediction[:,:,4])
        #Softmax the class scores
        prediction[:,:,5: 5 + self.configer.get_num_classes()] = t.sigmoid(prediction[:,:, 5 : 5 + self.configer.get_num_classes()])
        
        return prediction

###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

#from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
#    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
#from torch.nn import functional as F
#from torch.autograd import Variable
torch_ver = t.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(t.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = t.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = t.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(t.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = t.bmm(proj_query, proj_key)
        energy_new = t.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = t.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
