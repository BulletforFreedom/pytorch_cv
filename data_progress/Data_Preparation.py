
# coding: utf-8

# In[1]:


import xml.etree.ElementTree as ET
#import pickle
import os
#from os import listdir, getcwd
#from os.path import join
import glob
#import json
#import pdb


# In[7]:

#-----类别信息-----#
classes = []
for i in range(1,19,1):
    classes.append(str(i))

#classes_ch = ["雀巢丝滑拿铁","饮料","泡面","红酒","零食","矿泉水","酸奶"]


# In[5]:

#-----数据路径-----#

# 数据文件根目录
wd = "/mnt/disk1/lvsikai/missfresh/MissFreshSmartShelf_Exp/"

# raw images(*.jpg) dir
raw_images_dir = os.path.join(wd,'JPEGImages')

# pascal-voc labels(*.xml) dir
voc_labels_dir = os.path.join(wd,"Annotations")

# imageset dir(train.txt/val.txt)
imageset_dir = os.path.join(wd,'ImageSets','Main')


# In[4]:

'''划分训练集(training set)和验证集(validating set)'''

trainSet_list = []
valSet_list = []
testSet_list = []

val_train_ratio = [1,9] #验证集和训练集照片数量比例

## start
paths = glob.glob(os.path.join(raw_images_dir,'*.jpg'))
imgNum = len(paths)

tot = sum(val_train_ratio)
valNum = int(float(val_train_ratio[0])/tot*imgNum)
trainNum = imgNum-valNum

trainSet_list = paths[0:trainNum]
valSet_list = paths[trainNum:]

print("TrainNum:%d ,ValNum:%d"%(len(trainSet_list),len(valSet_list)))

#--save
trainf = open(os.path.join(imageset_dir,'train.txt'),'w')
for path in trainSet_list:
    trainf.write(path.split('/')[-1].strip()[0:-4]+'\n')
trainf.close()

valf = open(os.path.join(imageset_dir,'val.txt'),'w')
for path in valSet_list:
    valf.write(path.split('/')[-1].strip()[0:-4]+'\n')
valf.close()


# In[6]:

'''获取数据集绝对路径'''

dataset_dir = imageset_dir
image_dir = raw_images_dir
save_dir = wd

# --training set
train_set_file_path = os.path.join(dataset_dir,'train.txt')
f = open(train_set_file_path,'r')
name_list = f.readlines()
name_list = [x.strip() for x in name_list]
f.close()
#pdb.set_trace()
save_f = open(os.path.join(save_dir,'train.txt'),'w')
for name in name_list:
    save_f.write(image_dir+'/'+name+'.jpg'+'\n')
save_f.close()

# --val set
val_set_file_path = os.path.join(dataset_dir,'val.txt')
f = open(val_set_file_path,'r')
name_list = f.readlines()
name_list = [x.strip() for x in name_list]
f.close()
#pdb.set_trace()
save_f = open(os.path.join(save_dir,'val.txt'),'w')
for name in name_list:
    save_f.write(image_dir+'/'+name+'.jpg'+'\n')
save_f.close()


# In[8]:

'''Pascal-Voc Labels to DK Labels'''

sets=[('#', 'train'),('#','val')]


def convert(size, box):
    '''
    [topleft_x,topleft_y,bottomdown_x,bottomdown_y]->[center_x,center_y,w,h]
    '''
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(workingdir,voc_xml_path, image_id):
    in_file = open(voc_xml_path)
    out_file = open('%s/dk_labels/%s.txt'%(workingdir, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    #pdb.set_trace()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        
        #cls_utf = cls.encode('utf-8')
        #if cls_utf not in classes_ch or int(difficult)==1:
        #    continue
        #cls_id = classes_ch.index(cls_utf)
        
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
       
        print ("%s  %s %s"%(cls,image_id,cls_id))

        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    in_file.close()
    out_file.close()

## dk format label save path
dk_label_abs_dir = os.path.join(wd,'dk_labels')
if not os.path.exists(dk_label_abs_dir):
    os.makedirs(dk_label_abs_dir)
    
## pascal-voc format label file(*.xml) dir
voc_xml_dir = voc_labels_dir
voc_xml_paths = glob.glob(os.path.join(voc_xml_dir,'*.xml'))

## raw images(*.jpg)
image_dir = raw_images_dir
image_paths = glob.glob(os.path.join(image_dir,'*.jpg'))

## 检查标注数据是否和图片数据是否一一对应
diff=False

jpg_names = []
xml_names = []
miss = []
for imgpath in image_paths:
    name = imgpath.split('/')[-1].strip()[0:-4]
    jpg_names.append(name)
for xml in voc_xml_paths:
    xml_name = xml.split('/')[-1].strip()[0:-4]
    xml_names.append(xml_name)

for i in jpg_names:
    if i not in xml_names:
        miss.append(i)
        diff = True
if diff:
    print("file missing!")
    print (miss)
    exit()
else:
    pass

#pdb.set_trace()

## 开始转换主程序
for _,dataset in sets:
    set_path = os.path.join(wd,'ImageSets/Main/%s.txt'%dataset)
    infile = open(set_path,'r')
    names = infile.read().strip().split('\n')
    infile.close()

    for name in names:
        if name in jpg_names:
            img_path = os.path.join(wd,'JPEGImages/%s.jpg'%name)
        else:
            print("file missing in JPEGImages")
            print (name)
            exit()
        voc_xml_path = os.path.join(wd,'Annotations/%s.xml'%name)
        convert_annotation(wd,voc_xml_path,name)

    #pdb.set_trace()


# In[ ]:




# In[ ]:



