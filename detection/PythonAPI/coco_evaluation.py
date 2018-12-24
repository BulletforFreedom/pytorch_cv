import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import os

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm', 'bbox', 'keypoints']
annType = annType[1]  # specify type here
prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
print('Running demo for *%s* results.' % (annType))

# initialize COCO ground truth api
dataDir = '.'
dataType = 'gt_test'
annFile = '/home/lsk/Downloads/test_img/ImageSets/gt.json'
cocoGt = COCO(annFile)

# initialize COCO detections api
resFile = '/home/lsk/Downloads/eval_yolo_detection/results/predict/missfresh-yolo-voc-800-0724/10000/new.json'
predict_results_second_lever_dir = \
          '/home/lsk/Downloads/eval_yolo_detection/results/predict/missfresh-yolo-voc-800-0724/10000'
cocoDt = cocoGt.loadRes(resFile)

imgIds = sorted(cocoGt.getImgIds())
print(len(imgIds))
imgIds = imgIds[0:73]
# imgId = imgIds[np.random.randint(30)]

# running evaluation
cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = imgIds 
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()


result_f = open(os.path.join(predict_results_second_lever_dir,'mAP.txt'),'w')
result_f.write( 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' + str(cocoEval.stats[0]) + '\n')
result_f.write( 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = ' + str(cocoEval.stats[1]) + '\n')
result_f.write( 'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = ' + str(cocoEval.stats[2]) + '\n' + '\n')
result_f.write( 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = ' + str(cocoEval.stats[6]) + '\n')
result_f.write( 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = ' + str(cocoEval.stats[7]) + '\n')
result_f.write( 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' + str(cocoEval.stats[8]) + '\n')
result_f.close()
    
print(cocoEval.stats)
