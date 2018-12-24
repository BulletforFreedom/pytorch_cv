# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:01:59 2018

@author: lvsikai
"""

import numpy as np
import os

dataset_dir = "/home/lsk/Downloads/test_img"

# raw images(*.jpg) dir
raw_images_dir = os.path.join(dataset_dir,'JPEGImages')

# pascal-voc labels(*.xml) dir
voc_labels_dir = os.path.join(dataset_dir,"dk_labels")

# imageset dir(train.txt/val.txt)
imageset_dir = os.path.join(dataset_dir,'ImageSets','Main')

# 定义Box类，描述bounding box的坐标
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


# 计算两个box在某个轴上的重叠部分
# x1是box1的中心在该轴上的坐标
# len1是box1在该轴上的长度
# x2是box2的中心在该轴上的坐标
# len2是box2在该轴上的长度
# 返回值是该轴上重叠的长度
def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left


# 计算box a 和box b 的交集面积
# a和b都是Box类型实例
# 返回值area是box a 和box b 的交集面积
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area


# 计算 box a 和 box b 的并集面积
# a和b都是Box类型实例
# 返回值u是box a 和box b 的并集面积
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


# 计算 box a 和 box b 的 iou
# a和b都是Box类型实例
# 返回值是box a 和box b 的iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


def avg_iou(boxes, centroids):  
    n= len(boxes)  
    sums = 0.  
    for i in range(n):  
        # note IOU() will return array which contains IoU for each centroid and X[i]  
        # slightly ineffective, but I am too lazy  
        sums += max(iou(boxes[i], centroids))  
    return sums / n 

def iou(x, centroids):  
    dists = []  
    for centroid in centroids:  
        c_w = centroid.w
        c_h = centroid.h 
        w = x.w
        h = x.h
        if c_w >= w and c_h >= h:  
            dist = w * h / (c_w * c_h)  
        elif c_w >= w and c_h <= h:  
            dist = w * c_h / (w * h + (c_w - w) * c_h)  
        elif c_w <= w and c_h >= h:  
            dist = c_w * h / (w * h + c_w * (c_h - h))  
        else:  # means both w,h are bigger than c_w and c_h respectively  
            dist = (c_w * c_h) / (w * h)  
        dists.append(dist)  
    return np.array(dists)  

# 使用k-means ++ 初始化 centroids，减少随机初始化的centroids对最终结果的影响
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# 返回值centroids 是初始化的n_anchors个centroid
def init_centroids(boxes,n_anchors):
    centroids = []
    boxes_num = len(boxes)

    centroid_index = np.random.choice(boxes_num, 1)#随机选一个框
    centroids.append(boxes[centroid_index[0]])

    print(centroids[0].w,centroids[0].h)

    for centroid_index in range(0,n_anchors-1):

        sum_distance = 0
        distance_thresh = 0
        distance_list = []
        cur_sum = 0

        for box in boxes:
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = (1 - box_iou(box, centroid))
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance*np.random.random()

        for i in range(0,boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                print(boxes[i].w, boxes[i].h)
                break
    print(len(centroids))

    return centroids


# 进行 k-means 计算新的centroids
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# centroids是所有簇的中心
# 返回值new_centroids 是计算出的新簇中心
# 返回值groups是n_anchors个簇包含的boxes的列表
# 返回值loss是所有box距离所属的最近的centroid的距离的和
def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - box_iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= len(groups[i])
        new_centroids[i].h /= len(groups[i])

    return new_centroids, groups, loss


# 计算给定bounding boxes的n_anchors数量的centroids
# label_path是训练集列表文件地址
# n_anchors 是anchors的数量
# loss_convergence是允许的loss的最小变化值
# grid_size * grid_size 是栅格数量
# iterations_num是最大迭代次数
# plus = 1时启用k means ++ 初始化centroids
def compute_centroids(n_anchors,loss_convergence,grid_size,iterations_num,plus):

    boxes = []
    label_files = []
    f = open(os.path.join(imageset_dir,'val.txt'),'r')
    for line in f:
        if(len(line.strip())<27):
            label_files.append(os.path.join(voc_labels_dir+'/'+line.strip()+'.txt'))
    f.close()
    #print label_files
    for label_file in label_files:
        f = open(label_file)
        for line in f:
            temp = line.strip().split(" ")
            if len(temp) > 1:
                boxes.append(Box(0, 0, float(temp[3]), float(temp[4])))

    if plus:
        centroids = init_centroids(boxes, n_anchors)
    else:
        centroid_indices = np.random.choice(len(boxes), n_anchors)
        centroids = []
        for centroid_index in centroid_indices:
            centroids.append(boxes[centroid_index])
    
    # iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while (True):
        print(iterations)
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations += 1
        print("loss = %f" % loss)
        if abs(old_loss - loss) < loss_convergence or iterations > iterations_num:
            break        
        old_loss = loss

        for centroid in centroids:
            print(centroid.w * grid_size, centroid.h * grid_size)

    # print result
    for centroid in centroids:
        print("k-means result：")
        print(centroid.w, centroid.h)
        
    
    distance = avg_iou(boxes, centroids)
    print()
    print(distance)
    

if __name__ == "__main__":
    n_anchors = 5
    loss_convergence = 1e-6
    grid_size = 13
    iterations_num = 1000
    plus = 1
    compute_centroids(n_anchors,loss_convergence,grid_size,iterations_num,plus)
    #8  0.7919249143927691
    #7 0.7816997857344449
    #6 0.7710491678738098
    #5 0.7570055080291511
    #4 0.7415220181411415
    #3 0.7127174670732532
    #2 0.6613740144150089
    #12 0.8201601008635713