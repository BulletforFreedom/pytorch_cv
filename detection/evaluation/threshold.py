# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:23:10 2018

@author: lvsikai
"""

import os
import conf as cf
import utils as us
import matplotlib.pyplot as plt

def find_thres(dataset,model_name,test_info,num_class):
    dataset_file = os.path.join(cf.dataset_dir, 'ImageSets', 'Main', '%s.txt' % dataset)
    f = open(dataset_file)
    img_names = f.readlines()
    f.close()

    img_names = [x.strip() for x in img_names]

    # --get predict results
    predict_results_first_lever_dir = os.path.join(cf.wd,'results', 'predict', test_info)
    if not os.path.exists(predict_results_first_lever_dir):
        print("Folder not exist!")
        print predict_results_first_lever_dir
        exit()

    predict_results_second_lever_dir = os.path.join(predict_results_first_lever_dir, model_name)
    if not os.path.exists(predict_results_second_lever_dir):
        os.mkdir(predict_results_second_lever_dir)

    predict_results_dir = os.path.join(predict_results_second_lever_dir, dataset)
    if not os.path.exists(predict_results_dir):
        print("Folder not exist!")
        print predict_results_dir
        exit()

  
    count_scores = [[0]*19 for i in range(num_class)]
    
    for name in img_names:
        if( len(name)>27 or name[0]=='g'):
            continue
        # --get prediction
        predict_file = open(os.path.join(predict_results_dir,'%s.txt'%name))
        pt_list = predict_file.readlines()
        predict_file.close()
        pt_list = [x.strip().split(' ') for x in pt_list]
        pred_class_ids = [int(x[0]) for x in pt_list]
        pred_scores = [float(x[1]) for x in pt_list]
        count=0
        for x in pred_scores:
            if(x>0.95):
                count_scores[pred_class_ids[count]][18]+=1
            elif(x>0.9):
                count_scores[pred_class_ids[count]][17]+=1
            elif(x>0.85):
                count_scores[pred_class_ids[count]][16]+=1
            elif(x>0.8):
                count_scores[pred_class_ids[count]][15]+=1
            elif(x>0.75):
                count_scores[pred_class_ids[count]][14]+=1
            elif(x>0.7):
                count_scores[pred_class_ids[count]][13]+=1
            elif(x>0.65):
                count_scores[pred_class_ids[count]][12]+=1
            elif(x>0.6):
                count_scores[pred_class_ids[count]][11]+=1
            elif(x>0.55):
                count_scores[pred_class_ids[count]][10]+=1
            elif(x>0.5):
                count_scores[pred_class_ids[count]][9]+=1
            elif(x>0.45):
                count_scores[pred_class_ids[count]][8]+=1
            elif(x>0.4):
                count_scores[pred_class_ids[count]][7]+=1
            elif(x>0.35):
                count_scores[pred_class_ids[count]][6]+=1
            elif(x>0.3):
                count_scores[pred_class_ids[count]][5]+=1
            elif(x>0.25):
                count_scores[pred_class_ids[count]][4]+=1
            elif(x>0.2):
                count_scores[pred_class_ids[count]][3]+=1
            elif(x>0.15):
                count_scores[pred_class_ids[count]][2]+=1
            elif(x>0.1):
                count_scores[pred_class_ids[count]][1]+=1
            elif(x>0.05):
                count_scores[pred_class_ids[count]][0]+=1
            else:
                break
            count+=1
    return count_scores

if __name__ == "__main__":

    # {(model_name,dataset_name),...,...}
    # 模型文件(.weight)前缀
    ds_prefix = 'yolo-voc-800_'

    # 本次实验的名称(同一个模型可以用来做不同类型的实验)
    ds_test = 'missfresh-yolo-voc-800-0517'

    # perform object detection on these dataSets
    sets = ['test']

    # checkpoints of models
    checkpoints = [12000,20000,24000]

    DataSets = us.make_dataset(prefix=ds_prefix, test_info=ds_test, sets=sets, iterations=checkpoints)
    aaaa=[]
    num_class=26
    # get predict results
    for ds in DataSets:
        model_name = ds[0]
        dataset_name = ds[1]
        test_info = ds[3]

        if dataset_name=='test':
            aaaa.append(find_thres(dataset_name, model_name,test_info,num_class))
            #main_plot_bb_on_mix(dataset_name, model_name)
            #pass
        else:
            aaaa.append(find_thres(dataset_name, model_name,test_info,num_class))
    b=[]
    for i in range(19):
        b.append(i/20.+0.05)
    for i in range(num_class):
        count=0
        for ds in DataSets:
            plt.plot(b,aaaa[count][i],label=ds[0])
            count+=1
        plt.title('%d'%(i+1))
        plt.legend()
        plt.show()
