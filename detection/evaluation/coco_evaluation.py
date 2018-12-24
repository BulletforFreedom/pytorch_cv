import matplotlib.pyplot as plt
from PythonAPI.pycocotools.coco import COCO
from PythonAPI.pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
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
resFile = '/home/lsk/Downloads/eval_yolo_detection/results/predict/missfresh-yolo-voc-800-0709/16000/new.json'
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
print(cocoEval.stats)
