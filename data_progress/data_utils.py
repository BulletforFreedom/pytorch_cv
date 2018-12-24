#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:31:34 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""

from urllib.request import urlretrieve
import zipfile
import h5py
'''
output_directory='/home/lsk/Downloads/3d_pose_baseline_pytorch/data/'
h36m_path='/home/lsk/Downloads/3d_pose_baseline_pytorch/data/h36m/h3.6m.zip'
h36m_dataset_url = 'http://www.cs.stanford.edu/people/ashesh/h3.6m.zip'
urlretrieve(h36m_dataset_url, h36m_path)
print('Extracting Human3.6M dataset...')
with zipfile.ZipFile(h36m_path, 'r') as archive:
    archive.extractall(output_directory)
'''
'''
f=h5py.File('/home/lsk/Downloads/3d_pose_baseline_pytorch/data/h36m/annot.h5','r')
for key in f.keys():
    print(key)
    print(f[key].shape)
'''
f1=h5py.File('/home/lsk/Downloads/images/annot_train.h5','r')
for n, key1 in enumerate(sorted(f1.keys())):
    print(key1)
    print(f1[key1].shape)   
print('\n')
f2=h5py.File('/home/lsk/Downloads/images/annot_val.h5','r')
for key2 in f2.keys():
    print(key2)
    print(f2[key2].shape)
print('\n')
f1=h5py.File('/home/lsk/Downloads/h36m/images/h36m_annot/train.h5','r')
for n, key1 in enumerate(sorted(f1.keys())):
    print(key1)
    print(f1[key1].shape)   
print('\n')
f2=h5py.File('/home/lsk/Downloads/h36m/images/h36m_annot/valid.h5','r')
for key2 in f2.keys():
    print(key2)
    print(f2[key2].shape)
    
'''
train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d =\
data_utils.create_2d_data( actions, FLAGS.data_dir, rcams )

train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d,\
dim_to_use_3d, train_root_positions, test_root_positions =\
data_utils.read_3d_data(actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14 )

import tensorflow as tf

def isPalindrome(s):
    """
    :type s: str
    :rtype: bool
    """
    for c in string.punctuation:
        s = s.replace(c,'')
    s=s.lower().replace(' ','')
    for i in range(0,len(s)//2):
        if s[i]!=s[-1-i]:
            return 0;
    return 1;
    
import string
l=';aljf;lajf;qlfjam,.am./z'
m = l
for c in string.punctuation:
    m = m.replace(c,'')
'''