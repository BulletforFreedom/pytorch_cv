import json
import numpy as np
import evaluation as my_eval
import os
import sys
import time
from evaluation import fuckyou_trans


class load_data:
    """docstring for evaluation"""

    def __init__(self, file_path, is_gt=True, threshold=0.05):
        self.boxes = {}
        self.scores = {}
        self.class_ids = {}

        self.file_path = file_path
        self.threshold = threshold
        self.cat_num = 0

        if is_gt:
            self.fetch_data()
        else:
            self.fetch_data_pred()

    def fetch_data(self):

        with open(self.file_path) as f:
            js_format = json.load(f)
            # print(js_format)
            gt_anno = js_format['annotations']

        # store gt_bbox and gt_class_ids
        for item in gt_anno:
            #{'id': 1, 'iscrowd': 0, 'image_id': 1, 'category_id': 1, 'area': 0.0, 'segmentation': [[0.0]],
            #'bbox': [-1, 262, 252, 284]}
            print(item['category_id'])
            cate = fuckyou_trans(item['category_id'])
            print(cate)
            # if cate == 7:
            #     print(cate)
            self.cat_num = max(cate, self.cat_num)

            if item['image_id'] in self.class_ids:
                if cate in self.class_ids[item['image_id']]:
                    self.boxes[item['image_id']][cate] = np.concatenate(
                        (self.boxes[item['image_id']][cate], item['bbox']))
                    self.class_ids[item['image_id']][cate] = np.concatenate(
                        (self.class_ids[item['image_id']][cate], [cate]))
                else:
                    self.boxes[item['image_id']][cate] = np.array(
                        item['bbox'])
                    self.class_ids[item['image_id']][cate] = np.array(
                        [item['category_id']])
            else:
                self.boxes[item['image_id']] = {}
                self.boxes[item['image_id']
                           ][cate] = np.array(item['bbox'])
                self.class_ids[item['image_id']] = {}
                self.class_ids[item['image_id']][cate] = np.array(
                    [cate])
        # print(self.gt_boxes[1][1])
        for i in range(len(self.boxes)):
            for j in self.boxes[i + 1]:
                self.boxes[i + 1][j] = self.boxes[i +
                                                  1][j].reshape([-1, 4])
                # self.boxes[i +
                #            1][j] = self.format_trans(self.boxes[i + 1][j])
                # self.gt_class_ids[i +
                #                   1][j] = self.fuckyou_trans(self.gt_class_ids[i + 1][j])

    def fetch_data_pred(self):
        with open(self.file_path) as f:
            js_format = json.load(f)

        for item in js_format:
            # print(item)
            # break
            if item['score'] < self.threshold:
                continue

            if item['image_id'] in self.class_ids:
                if item['category_id'] in self.class_ids[item['image_id']]:
                    self.boxes[item['image_id']][item['category_id']] = np.concatenate(
                        (self.boxes[item['image_id']][item['category_id']], item['bbox']))
                    self.class_ids[item['image_id']][item['category_id']] = np.concatenate(
                        (self.class_ids[item['image_id']][item['category_id']], [item['category_id']]))
                    self.scores[item['image_id']][item['category_id']] = np.concatenate(
                        (self.scores[item['image_id']][item['category_id']], [item['score']]))
                else:
                    self.boxes[item['image_id']][item['category_id']] = np.array(
                        item['bbox'])
                    self.class_ids[item['image_id']][item['category_id']] = np.array(
                        [item['category_id']])
                    self.scores[item['image_id']][item['category_id']] = np.array([
                        item['score']])
            else:
                self.boxes[item['image_id']] = {}
                self.boxes[item['image_id']
                           ][item['category_id']] = np.array(item['bbox'])
                self.class_ids[item['image_id']] = {}
                self.class_ids[item['image_id']][item['category_id']] = np.array(
                    [item['category_id']])
                self.scores[item['image_id']] = {}
                self.scores[item['image_id']][item['category_id']] = np.array([
                    item['score']])

        for i in range(len(self.boxes)):
            try:
                for j in self.boxes[i + 1]:
                    self.boxes[i + 1][j] = self.boxes[i +
                                                      1][j].reshape([-1, 4])
                    # self.boxes[i +
                    #            1][j] = self.format_trans(self.boxes[i + 1][j])
            except:
                continue
        # print(self.pred_scores[2])
        # print('----------------------------')
        # print(self.gt_boxes[2])

    def switch_format(self):
        for i in range(len(self.boxes)):
            try:
                for j in self.boxes[i + 1]:

                    self.boxes[i +
                               1][j] = self.format_trans(self.boxes[i + 1][j])
            except:
                continue

    def format_trans(self, bbox_ratio_format):
        bbox_data_format = []
        for i in range(bbox_ratio_format.shape[0]):
            bbox_data_format.append([bbox_ratio_format[i][0], bbox_ratio_format[i][1], bbox_ratio_format[i][0] + float(
                bbox_ratio_format[i][2]), bbox_ratio_format[i][1] + float(bbox_ratio_format[i][3])])
        bbox_data_format = np.array(bbox_data_format)
        # print(bbox_data_format)
        return bbox_data_format


if __name__ == '__main__':
    a = load_data('data/Retinanet_size300/model_iter10499.json', is_gt=False)
    print(a.boxes[1][4])
