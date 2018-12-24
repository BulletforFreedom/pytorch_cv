# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 11:23:09 2018

@author: lvsikai
"""

import os
import codecs
from PIL import Image

dataset_dir = "C:\\Users\\lvsikai\\Desktop\\Yi+\\Missfresh\\4MissFreshSmartShelf_Exp"

# raw images(*.jpg) dir
raw_images_dir = os.path.join(dataset_dir,'JPEGImages')

# pascal-voc labels(*.xml) dir
voc_labels_dir = os.path.join(dataset_dir,"Annotation\\JPEGImages")

# pascal-voc labels(*.txt) dir
txt_labels_dir = os.path.join(dataset_dir,"test")#dk_labels

# imageset dir(train.txt/val.txt)
imageset_dir = os.path.join(dataset_dir,'ImageSets','Main')

#print('脚本名:' + sys.argv[0])
#print('转换文件:'+sys.argv[1])

dataset_file = os.path.join(imageset_dir, 'test.txt')
f = open(dataset_file)
file_names = f.readlines()
f.close()

file_names = [x.strip() for x in file_names]#.strip().split(' ')
#print len(file_names)

for file_name in file_names:
    if file_name!='-':
        print(file_name)
        Ann_file=os.path.join(voc_labels_dir, '%s.xml' % file_name)
        lbl_file=os.path.join('C:\\Users\\lvsikai\\Desktop\\repositories\\eval_yolo_detection\\results\\predict\\missfresh-yolo-voc-800-0518\\yolo-voc-800_24000\\test', '%s.txt' % file_name)
        img = Image.open(raw_images_dir+'\\%s.jpg' %file_name)
        img_w = img.size[0]
        img_h = img.size[1]
        with codecs.open(Ann_file,'w') as xml:
            xml.write('<?xml version="1.0"?>\n')
            xml.write('<annotation>\n')
            xml.write('<folder>Yi+</folder>\n')
            xml.write('<filename>'+file_name+'.jpg</filename>\n')
            xml.write('<source>\n')
            xml.write('<database>The Yi+ Database</database>\n')#未知
            xml.write('<annotation>VC Yi+</annotation>\n')
            xml.write('<image>Unknow</image>\n')
            xml.write('<flickrid>Unknow</flickrid>\n')
            xml.write('</source>\n')
            xml.write('<owner>\n')
            xml.write('<flickrid>Unknow</flickrid>\n')
            xml.write('<name>Unknow</name>\n')
            xml.write('</owner>\n')
            xml.write('<size>\n')
            xml.write('<width>'+str(img_w)+'</width>\n')
            xml.write('<height>'+str(img_h)+'</height>\n')
            xml.write('<depth>3</depth>')
            xml.write('</size>\n')
            xml.write('<segmented>0</segmented>\n')
            with codecs.open(lbl_file,'r') as f:
                dataset = f.readlines()
            dataset = [x.strip().split(' ') for x in dataset]
            for data in dataset:
                if(float(data[1])>.6):
                    #print data[1]
                    xml.write('<object>')
                    xml.write('<name>'+str(int(data[0])+1)+'</name>\n')
                    xml.write('<pose>Left</pose>\n')
                    xml.write('<truncated>0</truncated>\n')
                    xml.write('<difficult>0</difficult>\n')
                    xml.write('<bndbox>\n')
                    b_w = float(data[4])*img_w
                    b_h = float(data[5])*img_h
                    c_x = float(data[2])*img_w
                    c_y = float(data[3])*img_h
                    xml.write('<xmin>'+str(int(max([0, (c_x - 0.5 * b_w) ])))+'</xmin>\n')
                    xml.write('<ymin>'+str(int(max([0, (c_y - 0.5 * b_h) ])))+'</ymin>\n')
                    xml.write('<xmax>'+str(int(min([img_w, (c_x + 0.5 * b_w) ])))+'</xmax>\n')
                    xml.write('<ymax>'+str(int(min([img_h, (c_y + 0.5 * b_h) ])))+'</ymax>\n')
                    xml.write('</bndbox>\n')
                    xml.write('</object>\n')
            xml.write('</annotation>')