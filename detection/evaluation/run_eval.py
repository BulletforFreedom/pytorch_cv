import json
import numpy as np
import evaluation as my_eval
import os
import sys
import time
from evaluation import fuckyou_trans


class evaluation:
    """docstring for evaluation"""

    def __init__(self, gt_file, pred_file, cat=False, maxDets=100, threshold=0.05, iou_threshold=0.5):
        self.pred_boxes = {}
        self.pred_scores = {}
        self.pred_class_ids = {}
        self.gt_boxes = {}
        self.gt_class_ids = {}

        self.pred_boxes_cat = {}
        self.pred_scores_cat = {}
        self.pred_class_ids_cat = {}
        self.gt_boxes_cat = {}
        self.gt_class_ids_cat = {}

        self.gt_file = gt_file
        self.pred_file = pred_file
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.cat_flag = cat
        self.pred_match_cat = {}
        self.scores_cat = {}
        self.cat_num = 0
        self.maxDets = maxDets
        self.eval = {}
        if cat == False:
            self.fetch_data()
        else:
            self.fetch_data_cat()

    def fetch_data(self):
        print('loading annotations into memory......(ignoring category_info)')
        t = time.time()
        self.eval = {'FP_cls': 0, 'FP_bg': 0, 'FN': 0}
        with open(self.gt_file) as f:
            js_format = json.load(f)
            gt_anno = js_format['annotations']

        # store gt_bbox and gt_class_ids
        for item in gt_anno:
            cate = fuckyou_trans(item['category_id'])
            # if cate == 7:
            #     print(cate)
            self.cat_num = max(cate, self.cat_num)
            if item['image_id'] in self.gt_class_ids:
                self.gt_boxes[item['image_id']] = np.concatenate(
                    (self.gt_boxes[item['image_id']], item['bbox']))
                self.gt_class_ids[item['image_id']] = np.concatenate(
                    (self.gt_class_ids[item['image_id']], [cate]))
            else:
                self.gt_boxes[item['image_id']] = np.array(item['bbox'])
                self.gt_class_ids[item['image_id']] = np.array(
                    [cate])
        for i in range(len(self.gt_boxes)):
            self.gt_boxes[i + 1] = self.gt_boxes[i + 1].reshape([-1, 4])
            self.gt_boxes[i + 1] = self.format_trans(self.gt_boxes[i + 1])
            # self.gt_class_ids[i +
            #                   1] = self.fuckyou_trans(self.gt_class_ids[i + 1])

        with open(self.pred_file) as f:
            js_format = json.load(f)

        for item in js_format:
            # print(item)
            # break
            if item['score'] < self.threshold:
                continue
            if item['image_id'] in self.pred_class_ids:
                # print(item['bbox'])
                self.pred_boxes[item['image_id']] = np.concatenate(
                    (self.pred_boxes[item['image_id']], item['bbox']))
                self.pred_class_ids[item['image_id']] = np.concatenate(
                    (self.pred_class_ids[item['image_id']], [item['category_id']]))
                self.pred_scores[item['image_id']] = np.concatenate(
                    (self.pred_scores[item['image_id']], [item['score']]))
                # if item['image_id'] == 1:
                #     print(item['category_id'])
                # print(' ')
            else:
                self.pred_boxes[item['image_id']] = np.array(item['bbox'])
                self.pred_class_ids[item['image_id']
                                    ] = np.array([item['category_id']])
                # if item['image_id'] == 1:
                #     print(item['category_id'])
                # pred_name['image_id'] = item
                self.pred_scores[item['image_id']] = np.array([item['score']])
        for i in range(len(self.pred_boxes)):
            try:
                self.pred_boxes[i + 1] = self.pred_boxes[i +
                                                         1].reshape([-1, 4])
                self.pred_boxes[i +
                                1] = self.format_trans(self.pred_boxes[i + 1])
            except:
                continue
        print("loading finished (t=%fs)" % (time.time() - t))

    def fetch_data_cat(self):
        print('loading annotations into memory......(category_info is considered)')
        t = time.time()
        self.eval = {'FP_cls': np.zeros([self.cat_num + 1]), 'FP_bg': np.zeros(
            [self.cat_num + 1]), 'FN': np.zeros([self.cat_num + 1])}
        with open(self.gt_file) as f:
            js_format = json.load(f)
            gt_anno = js_format['annotations']

        # store gt_bbox and gt_class_ids
        for item in gt_anno:
            #{'id': 1, 'iscrowd': 0, 'image_id': 1, 'category_id': 1, 'area': 0.0, 'segmentation': [[0.0]],
            #'bbox': [-1, 262, 252, 284]}

            cate = item['category_id']
            #cate = item['category_id']

            # if cate == 7:
            #     print(cate)
            self.cat_num = max(cate, self.cat_num)

            if item['image_id'] in self.gt_class_ids_cat:
                if cate in self.gt_class_ids_cat[item['image_id']]:
                    self.gt_boxes_cat[item['image_id']][cate] = np.concatenate(
                        (self.gt_boxes_cat[item['image_id']][cate], item['bbox']))
                    self.gt_class_ids_cat[item['image_id']][cate] = np.concatenate(
                        (self.gt_class_ids_cat[item['image_id']][cate], [cate]))
                else:
                    self.gt_boxes_cat[item['image_id']][cate] = np.array(
                        item['bbox'])
                    self.gt_class_ids_cat[item['image_id']][cate] = np.array(
                        [item['category_id']])
            else:
                self.gt_boxes_cat[item['image_id']] = {}
                self.gt_boxes_cat[item['image_id']
                                  ][cate] = np.array(item['bbox'])
                self.gt_class_ids_cat[item['image_id']] = {}
                self.gt_class_ids_cat[item['image_id']][cate] = np.array(
                    [cate])
        # print(self.gt_boxes[1][1])
        for i in range(len(self.gt_boxes_cat)):
            for j in self.gt_boxes_cat[i + 1]:
                self.gt_boxes_cat[i + 1][j] = self.gt_boxes_cat[i +
                                                                1][j].reshape([-1, 4])
                self.gt_boxes_cat[i +
                                  1][j] = self.format_trans(self.gt_boxes_cat[i + 1][j])
                # self.gt_class_ids[i +
                #                   1][j] = self.fuckyou_trans(self.gt_class_ids[i + 1][j])

        with open(self.pred_file) as f:
            js_format = json.load(f)
        i = 0
        for item in js_format:
            if item['score'] < self.threshold:
                continue

            if item['image_id'] in self.pred_class_ids_cat:
                if item['category_id'] in self.pred_class_ids_cat[item['image_id']]:
                    self.pred_boxes_cat[item['image_id']][item['category_id']] = np.concatenate(
                        (self.pred_boxes_cat[item['image_id']][item['category_id']], item['bbox']))
                    self.pred_class_ids_cat[item['image_id']][item['category_id']] = np.concatenate(
                        (self.pred_class_ids_cat[item['image_id']][item['category_id']], [item['category_id']]))
                    self.pred_scores_cat[item['image_id']][item['category_id']] = np.concatenate(
                        (self.pred_scores_cat[item['image_id']][item['category_id']], [item['score']]))
                else:
                    self.pred_boxes_cat[item['image_id']][item['category_id']] = np.array(
                        item['bbox'])
                    self.pred_class_ids_cat[item['image_id']][item['category_id']] = np.array(
                        [item['category_id']])
                    self.pred_scores_cat[item['image_id']][item['category_id']] = np.array([
                        item['score']])
            else:
                self.pred_boxes_cat[item['image_id']] = {}
                self.pred_boxes_cat[item['image_id']
                                    ][item['category_id']] = np.array(item['bbox'])
                self.pred_class_ids_cat[item['image_id']] = {}
                self.pred_class_ids_cat[item['image_id']][item['category_id']] = np.array(
                    [item['category_id']])
                self.pred_scores_cat[item['image_id']] = {}
                self.pred_scores_cat[item['image_id']][item['category_id']] = np.array([
                    item['score']])

        for i in range(len(self.pred_boxes_cat)):
            try:
                for j in self.pred_boxes_cat[i + 1]:
                    self.pred_boxes_cat[i + 1][j] = self.pred_boxes_cat[i +
                                                                        1][j].reshape([-1, 4])
                    self.pred_boxes_cat[i +
                                        1][j] = self.format_trans(self.pred_boxes_cat[i + 1][j])
            except:
                continue
        self.pred_match_cat = {i + 1: [] for i in range(self.cat_num)}
        self.scores_cat = {i + 1: [] for i in range(self.cat_num)}
        # print(self.pred_scores[2])
        # print('----------------------------')
        # print(self.gt_boxes[2])
        print("loading finished (t=%fs)" % (time.time() - t))

    def format_trans(self, bbox_ratio_format):
        bbox_data_format = []
        for i in range(bbox_ratio_format.shape[0]):
            bbox_data_format.append([bbox_ratio_format[i][0], bbox_ratio_format[i][1], bbox_ratio_format[i][0] + float(
                bbox_ratio_format[i][2]), bbox_ratio_format[i][1] + float(bbox_ratio_format[i][3])])
        bbox_data_format = np.array(bbox_data_format)
        # print(bbox_data_format)
        return bbox_data_format

    def get_mAP(self):
        a = 0
        if self.cat_flag == False:
            t = time.time()
            print('Runing per image evaluation........')
            for i in range(len(self.pred_boxes)):
                try:
                    per_image = my_eval.compute_map(self.gt_boxes[i + 1], self.gt_class_ids[i + 1],
                                                    self.pred_boxes[i + 1], self.pred_class_ids[i +
                                                                                                1], self.pred_scores[i + 1],
                                                    iou_threshold=self.iou_threshold)
                except:
                    continue

                b = per_image[0]
                F = per_image[1]
                self.eval['FP_bg'] += F['FP_bg']
                self.eval['FP_cls'] += F['FP_cls']
                self.eval['FN'] += F['FN']
                # if b < 0.8:
                #     print("Image:%d hasn't been fed up =.=" % i)
                a += b
            print('Accumulating evaluation results........')
            a = a / len(self.pred_boxes)
            self.eval['mAP'] = a
            print('Done (t=%fs).' % (time.time() - t))
            print(
                "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=all ] = %f" % a)
        else:
            IoU = np.arange(0.5, 1, 0.05)
            AP = np.zeros([10])
            t = time.time()
            print('Runing per class evaluation........')
            for i in range(10):
                try:
                    AP[i] = my_eval.compute_map_cat(self.gt_boxes_cat, self.pred_boxes_cat,
                                                    self.pred_scores_cat, self.cat_num, IoU[i])
                except:
                    print('AP[%d] computation fails' % i)
            print('Accumulating evaluation results........')
            print('Done (t=%fs).' % (time.time() - t))
            mAP = np.mean(AP)
            self.eval['mAP'] = mAP
            self.eval['mAP50'] = AP[0]
            self.eval['mAP75'] = AP[5]
            print(
                'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=all ] = %f' % mAP)
            print(
                'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=all ] = %f' % AP[0])
            print(
                'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=all ] = %f' % AP[5])

    def err_analysis(self):  # unfinished
        print('Runing error analysis......')
        t = time.time()
        if self.cat_flag == True:
            self.fetch_data()
        for i in range(len(self.pred_boxes)):
            per_image = my_eval.compute_map(self.gt_boxes[i + 1], self.gt_class_ids[i + 1],
                                            self.pred_boxes[i + 1], self.pred_class_ids[i +
                                                                                        1], self.pred_scores[i + 1],
                                            iou_threshold=self.iou_threshold)
            b = per_image[0]
            F = per_image[1]
            self.eval['FP_bg'] += F['FP_bg']
            self.eval['FP_cls'] += F['FP_cls']
            self.eval['FN'] += F['FN']
        print('Analysis finished (t=%fs)' % (time.time() - t))
        print('Attention : threshold was designed as : %f' % self.threshold)
        print('Number of FP_cls : %d' % self.eval['FP_cls'])
        print('Number of FP_bg : %d' % self.eval['FP_bg'])  
        print('Number of FN : %d' % self.eval['FN'])


if __name__ == "__main__":

    # a = evaluation('data/instances_gt_test.json',
    #                './data/Retinanet_unresized/result_9999.json', threshold=0.5, cat=True)
    # a.err_analysis()
    test = evaluation('/home/lsk/Downloads/test_img/ImageSets/gt.json',
                      '/home/lsk/Downloads/eval_yolo_detection/results/predict/missfresh-yolo-voc-800-0710/26000/new.json', threshold=0.5, cat=True)
    test.get_mAP()
