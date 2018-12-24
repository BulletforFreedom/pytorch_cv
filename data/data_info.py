#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:30:42 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""

education={"name_seq": ["read", "nap", "None", "stand_up", "hands_up"],
           "mean": [],
           "std": []
}
coco2014={"name_seq": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                   "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
                   "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                   "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                   "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                   "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                   "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                   "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                   "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
                   "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                   "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                   "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
           "mean": [0.485, 0.456, 0.406],
           "std": [0.229, 0.224, 0.225],
           "data_dir": '/home/lsk/Downloads/yolov3_pytorch/data/COCO_DET'
}