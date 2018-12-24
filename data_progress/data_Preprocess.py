#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:20:45 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""
import os
import glob
import cv2
import shutil
import random
import numpy as np

class data_Preprocess(object):
    
    @staticmethod
    def cp_images(orig_path, dest_path):
        '''
        copy images which are needed
        '''
        orig_dirs=glob.glob(os.path.join(orig_path, '*.jpg'))
        for idx, orig_path in enumerate(orig_dirs):
            name=orig_path.split('/')[-1].split('.')[0]            
            im = cv2.imread(orig_path)
            name=name+'-'+ str(idx)+'%'+str(len(orig_dirs))
            cv2.imshow(name,im)
            cv2.moveWindow(name,10,10)
            k = cv2.waitKey(0)
            if k==ord('q'):
                cv2.destroyAllWindows()
                break            
            elif k==ord('v'):
                shutil.copyfile(orig_path, os.path.join(dest_path, name + '.jpg'))
                cv2.destroyWindow(name)
            else:
                cv2.destroyWindow(name)
                
    @staticmethod
    def label_image(data_dir):        
        project_name='train_set'
        project_path=os.path.join(data_dir, project_name)
        class_dirs=[i for i in glob.glob(os.path.join(project_path, '*')) if len(i.split('.'))==1]
        label_dir=os.path.join(project_path, 'image_paths.txt')
        f_w=open(label_dir,'w')
        
        for label, class_dir in enumerate(class_dirs):
            img_paths=glob.glob(os.path.join(class_dir, '*.jpg'))
            for img_path in img_paths:
                f_w.write('%s$%d\n'%(img_path, label))            
        f_w.close()
    
    @staticmethod
    def partition_data(data_dir):
        trainSet_list = []
        valSet_list = []
        
        val_train_ratio = [2,8] #验证集和训练集照片数量比例
        
        ## start
        project_name='train_set'
        label_paths=os.path.join(data_dir, project_name, 'image_paths.txt')
        f = open(label_paths,'r')
        label_paths = f.readlines()
        f.close()
        imgNum = len(label_paths)
        random.shuffle(label_paths)
        tot = sum(val_train_ratio)
        valNum = int(float(val_train_ratio[0])/tot*imgNum)
        trainNum = imgNum-valNum
        
        trainSet_list = label_paths[0:trainNum]
        valSet_list = label_paths[trainNum:]
        
        print("TrainNum:%d ,ValNum:%d"%(len(trainSet_list),len(valSet_list)))
        
        #--save
        imageset_dir=os.path.join(data_dir, project_name)
        trainf = open(os.path.join(imageset_dir,'train.txt'),'w')
        for path in trainSet_list:
            trainf.write(path)
        trainf.close()
        
        valf = open(os.path.join(imageset_dir,'val.txt'),'w')
        for path in valSet_list:
            valf.write(path)
        valf.close()
        
    @staticmethod
    def read_video(video_path):
        video_path='/home/lsk/Downloads/pytorch_cv/data/education/videos/1.MOV'
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        while ret:            
            print(frame.shape)
            # show a frame
            cv2.imshow("capture", frame)
            ret, frame = cap.read()
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    
    @staticmethod
    def crop_and_save(img_path, results_path, write_path):
        img_dirs=glob.glob(os.path.join(img_path,'*'))
        for img_dir in img_dirs:
            img_name=img_dir.split('/')[-1].split('.')[0]
            image= cv2.imread(img_dir)
            box = open(os.path.join(results_path ,img_name + '.txt'))
            boxn = box.readlines()   
            box.close()
            arr = image.shape            
            width = arr[1]
            height = arr[0]
            len_box = len(boxn)
            for i in range(0,len_box):
                box2=boxn[i].split()
                box2[1]=float(box2[2])
                box2[2]=float(box2[3])
                box2[3]=float(box2[4])*1.2
                box2[4]=float(box2[5])*1.2
                box1 = (max([0, width*box2[1]-width*box2[3]/2]),
                        max([0, height*box2[2]-height*box2[4]/2]),
                        min([width, width*box2[3]]), 
                        min([height, height*box2[4]]))
            
                array_box1 = np.array(box1)
                array_box1 = array_box1.astype(int)
        #        cv2.rectangle(image,(array_box1[0],array_box1[1]),(array_box1[0]+array_box1[2],array_box1[1]+array_box1[3]),(0,255,0),3,8,0)
        #        cv2.putText(image,('%s'%(int(box2[0])+1)),(int(box1[0])+5,int(box1[1])+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,255),thickness=2)
                roii = image[array_box1[1]:array_box1[1]+array_box1[3], array_box1[0]:array_box1[0]+array_box1[2]]
                cv2.imwrite(write_path+'/'+img_name+'_'+str(i)+'.jpg',roii)
    
if __name__=='__main__':
    '''
    orig_path='/home/lsk/Downloads/education/images/label1213/crop/read'
    dest_path='/home/lsk/Downloads/pytorch_cv/data/education/train_set/None'
    data_Preprocess.cp_images(orig_path, dest_path)
    '''
    data_dir='/home/lsk/Downloads/pytorch_cv/data/education'
    data_Preprocess.label_image(data_dir)
    data_Preprocess.partition_data(data_dir)
    