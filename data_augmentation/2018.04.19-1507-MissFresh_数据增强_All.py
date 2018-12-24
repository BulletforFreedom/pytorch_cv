#coding:utf-8 允许中文注释
import numpy as np
import cv2
import math
import glob
import os
#from math import *
#image_path='C:\\Users\\Administrator\\Desktop\\23.jpg'
#image= cv2.imread(image_path)


wd = 'C:\\Users\\lvsikai\\Desktop\\Yi+\\Missfresh\\7MissFreshSmartShelf_Exp'

def Overlap_Rate(p1x,p1y,p2x,p2y,p3x,p3y,p4x,p4y):
    #p3、p4分别为groundtruth框的左上和右下角点，p1、p2为检测框的
    over_rate=0
    over_p1x=0
    over_p1y=0
    over_p2x=0
    over_p2y=0
#    area_gt=(p4y-p3y)*(p4x-p3x)#groundtruth框的面积
    area_test=(p2y-p1y)*(p2x-p1x)#检测框的面积
    if((p2y>p3y)and(p1y<p4y)and(p2x>p3x)and(p1x<p4x)):#如果有重合
        over_p1x=max(p1x,p3x)#重叠区左上角点
        over_p1y=max(p1y,p3y)
        over_p2x=min(p2x,p4x)#重叠区右下角点
        over_p2y=min(p2y,p4y)
        area_over=(over_p2y-over_p1y)*(over_p2x-over_p1x)
        over_rate=area_over/area_test#重合区面积比两框并集的面积
#        print("U并区面积：",area_gt)
#        print("重合区域面积：",area_over)
    return over_rate


def randomCrop(name, num_roi):  
    """
    num_roi为ROI区域总个数,num_sroi为小ROI区域的个数，总共定义5个大的ROI，还要定义num_sroi个小的ROI
    random_roi_存的四个值依次为左上角点横坐标、左上角点纵坐标、roi的宽、roi的高，lu=left_up,r=right,b=bottom
    截取原图的左上，右上，左下，右下，中间共五个ROI区域
    """
    image = cv2.imread(wd + '\\JPEGImages\\' + name + '.jpg') 
    box = np.loadtxt(wd + '\\dk_labels\\' + name + '.txt')
    print(image.shape)
    print('box_type:',type(box))
    
    num_sroi = num_roi - 5
    image_size = image.shape
    image_height = image_size[0]
    image_width = image_size[1]
    half_height = image_height >> 1
    half_width = image_width >> 1
    #math.ceil向上取整
    max_roi_h = math.ceil((image_height*2)/3)
    max_roi_w = math.ceil((image_width*2)/3)
    
    roi_h = np.random.randint(half_height, max_roi_h) 
    roi_w = np.random.randint(half_width, max_roi_w) 
    

    random_roi_lu = (0, 0 , roi_w, roi_h)
    random_roi_ru = (image_width - roi_w-1, 0, roi_w, roi_h)#减1防止ROI超出图片范围
    random_roi_lb = (0, image_height - roi_h -1, roi_w, roi_h)
    random_roi_rb = (image_width - roi_w-1, image_height - roi_h -1, roi_w, roi_h)
    random_roi_center = ((image_width - roi_w-1) >> 1, (image_height - roi_h -1) >> 1, roi_w, roi_h)
    
    random_roi = np.zeros((num_roi,4))
    random_roi[0] = random_roi_lu
    random_roi[1] = random_roi_ru
    random_roi[2] = random_roi_lb
    random_roi[3] = random_roi_rb
    random_roi[4] = random_roi_center
    
    #make txt and image dir
    crop_txt_dir = os.path.join(wd,'predit', 'txt')
    if not os.path.exists(crop_txt_dir):
        os.mkdir(crop_txt_dir)
    crop_image_dir = os.path.join(wd,'predit', 'image')
    if not os.path.exists(crop_image_dir):
        os.mkdir(crop_image_dir)
        
    #随机生成num_sroi个尺寸在原图的1/3到1/2之间的ROI区块
    thir_h = image_height/3
    thir_w = image_width/3
    for i in range(0,num_sroi):
        sroi_h = np.random.randint(thir_h, half_height) 
        sroi_w = np.random.randint(thir_w, half_width)
        lu_h = np.random.randint(0, image_height - sroi_h)
        lu_w = np.random.randint(0, image_width - sroi_w)
        random_roi[4+i+1] = (lu_w, lu_h, sroi_w-1, sroi_h-1)
    #数组整体转整型    
    random_roi = random_roi.astype(int)
    a=random_roi
    
    
    #image切分成num_roi张小图
    for i in range(0,num_roi):
        str_i = str(i+1)
        str_a='00'
        str_i=str_a+str_i
        str_i=str_i[-3:]
#        roii = image[a[i][1]:a[i][1]+a[i][3],a[i][0]:a[i][0]+a[i][2]]
        box_roi = a[i]#box_roi就是roii，存储形式为(x,y,w,h) xy为左上角点坐标
        print('box_roi:',box_roi)
        len_box = len(box)
        lst_all=[]
        result_txt_path = os.path.join(crop_txt_dir,'%s.txt'%(name+'_'+str_i))
        out_file = open(result_txt_path, 'w')
        for j in range(0,len_box):
            #单独卡粑粑柑阈值
            threshold_t=.3
            if box[j][0]==24 or box[j][0]==25:
                threshold_t=0.7
            #把框框相对于全图的坐标改为相对于ROI的坐标：(x,y,w,h) xy为中心点坐标
            lstj = [box[j][0],(box[j][1]*image_width-a[i][0])/a[i][2],(box[j][2]*image_height-a[i][1])/a[i][3],box[j][3]*image_width/a[i][2],box[j][4]*image_height/a[i][3]]
#            lstj=lst.tolist()
            #把box的中点xy坐标改为box1的左上角点xy坐标
            box1 = (image_width*box[j][1]-image_width*box[j][3]/2, image_height*box[j][2]-image_height*box[j][4]/2, 
                    image_width*box[j][3], image_height*box[j][4])
            array_box1 = np.array(box1)
            array_box1 = array_box1.astype(int)
            over_rate = Overlap_Rate(array_box1[0],array_box1[1],array_box1[0]+array_box1[2],array_box1[1]+array_box1[3],
                                     box_roi[0],box_roi[1],box_roi[0]+box_roi[2],box_roi[1]+box_roi[3])
            if(over_rate>threshold_t):
                #下面判断框的范围有没有超出ROI的范围：
                if(box1[0]<box_roi[0]):#box框的左上角点x坐标 在 ROI区域左上角点x坐标的左侧（超出ROI范围）(判断左边的x坐标)
                    lstj[1]=((box1[0]+box1[2]-box_roi[0])/2+2)/box_roi[2]#因为算的是中心点坐标，所以要除以2
                    lstj[3]=(box1[0]+box1[2]-box_roi[0])/box_roi[2]
                    
                if(box1[0]+box1[2]>box_roi[0]+box_roi[2]):#判断右边的x坐标:
                    lstj[1]=((box_roi[0]+box_roi[2]-box1[0])/2+box1[0]-box_roi[0]-2)/box_roi[2]
                    lstj[3]=(box_roi[0]+box_roi[2]-box1[0])/box_roi[2]
                    
                if(box1[1]<box_roi[1]):#判断上边y坐标
                    lstj[2]=((box1[1]+box1[3]-box_roi[1])/2+2)/box_roi[3]
                    lstj[4]=(box1[1]+box1[3]-box_roi[1])/box_roi[3]
                    
                if(box1[1]+box1[3]>box_roi[1]+box_roi[3]):#判断下边y坐标
                    lstj[2]=((box_roi[1]+box_roi[3]-box1[1])/2+box1[1]-box_roi[1]-2)/box_roi[3]
                    lstj[4]=(box_roi[1]+box_roi[3]-box1[1])/box_roi[3]
                    
                lst_all.append(lstj)
                out_file.write((str(int(box[j][0])) + " " + str(lstj[1]) + " " + str(lstj[2]) + " " + str(lstj[3]) + " " + str(lstj[4])) + '\n')
            else:
                print over_rate
                

#                cv2.rectangle(image,(array_box1[0],array_box1[1]),(array_box1[0]+array_box1[2],
#                              array_box1[1]+array_box1[3]),(0,255,255),3,8,0)
                
        #这一步是为了每次在image对应的ROI上画框之后，把image重置，防止本次画框对下次画框造成影响，（是在image上画框然后截取的ROI区域，而不是直接再ROI上面画框）
        roii = image[a[i][1]:a[i][1]+a[i][3],a[i][0]:a[i][0]+a[i][2]]

        result_image_path = os.path.join(crop_image_dir,'%s.jpg'%(name+'_'+str_i))
        cv2.imwrite(result_image_path,roii)
        out_file.close()
        break

    
def randomflip(name):
    random_num=np.random.randint(0,3)
    img = cv2.imread(wd + 'predit\\image\\' + name + '.jpg')
#    box = np.loadtxt(wd + 'results\\txt\\' + name + '.txt')
    box = open(os.path.join(wd + 'predit\\txt\\' + name + '.txt'))
    box1 = box.readlines()

    
    
    height,width=img.shape[:2]
    
    if random_num==0:
        degree=-90
    elif random_num==1:
        degree=90
    else:
        degree=0
    print(degree)
    #旋转后的尺寸
    heightNew=int(width*math.fabs(math.sin(math.radians(degree)))+height*math.fabs(math.cos(math.radians(degree))))
    widthNew=int(height*math.fabs(math.sin(math.radians(degree)))+width*math.fabs(math.cos(math.radians(degree))))
    
    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    
    matRotation[0,2] +=(widthNew-width)/2  #重点在这步，目前不懂为什么加这步
    matRotation[1,2] +=(heightNew-height)/2  #重点在这步
    
    imgRotation=cv2.warpAffine(img,matRotation,(widthNew,heightNew),borderValue=(255,255,255))
    
    #下面针对box
    len_box = len(box1)
    #对于没有标注数据的空图片直接跳过
    if(len_box==0):
        return
    print('len_box:',len_box)

    #make txt and image dir
    flip_txt_dir = os.path.join(wd,'predit', 'flip_txt')
    if not os.path.exists(flip_txt_dir):
        os.mkdir(flip_txt_dir)
    flip_image_dir = os.path.join(wd,'predit', 'flip_image')
    if not os.path.exists(flip_image_dir):
        os.mkdir(flip_image_dir)
        
    result_txt_path = os.path.join(flip_txt_dir,'%s.txt'%(name+'_'+'flip'))
    out_file = open(result_txt_path, 'w')
    for i in range(0,len_box):
        box2=box1[i].split()
        box_x,box_y,box_w,box_h=box2[1],box2[2],box2[3],box2[4]
        flip_box_x,flip_box_y,flip_box_w,flip_box_h=box_x,box_y,box_w,box_h
        if degree==90:
            flip_box_x=float(box_y)
            flip_box_y=1-float(box_x)
            flip_box_w=float(box_h)
            flip_box_h=float(box_w)
        elif degree==-90:
            flip_box_x=1-float(box_y)
            flip_box_y=float(box_x)
            flip_box_w=float(box_h)
            flip_box_h=float(box_w)

        out_file.write((str(int(box2[0])) + " " + str(flip_box_x) + " " + str(flip_box_y) + " " + str(flip_box_w) + " " + str(flip_box_h)) + '\n')
    
    out_file.close()
    result_image_path = os.path.join(flip_image_dir,'%s.jpg'%(name+'_'+'flip'))
    cv2.imwrite(result_image_path,imgRotation)
#    cv2.imshow("imgRotation",imgRotation)
#    cv2.waitKey(0)


image_paths = glob.glob(os.path.join(wd,'JPEGImages','*.jpg'))# 星号表示所有具有jpg后缀的文件
if __name__ == "__main__":
#    randomCrop('MissFresh_0408_train_0068',10)
   
    FLAG='crop'
    
    for imgpath in image_paths:
        if FLAG=='flip':
            name = imgpath.split('\\')[-1].strip()[0:-4]
            print('name:',name)
            randomflip(name)
        elif FLAG=='crop':
            name = imgpath.split('\\')[-1].strip()[0:-4]
            print('name:',name)
            randomCrop(name,10)
        break