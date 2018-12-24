#coding:utf-8 ÔÊÐíÖÐÎÄ×¢ÊÍ
import re
import requests
import os 
import sys



wds=['D:\\Landmark-building\\China\\11.09\\Xianggang\\Ruxin_Square\\',
     'D:\\Landmark-building\\China\\11.09\\Xianggang\\International_Finance_Center_Phase_II\\',
     'D:\\Landmark-building\\China\\11.09\\Shenzhen\\Shenzhen_International_Trade_Building\\',
     'D:\\Landmark-building\\China\\11.09\\Shenzhen\\Shenzhen_Electronics_Building\\',
     'D:\\Landmark-building\\China\\11.09\\Shenzhen\\Shenzhen_Shanghai_Hotel\\'
     ]

urls=['https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1541991800288_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E5%A6%82%E5%BF%83%E5%B9%BF%E5%9C%BA',
      'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1541991863222_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E5%9B%BD%E9%99%85%E9%87%91%E8%9E%8D%E4%B8%AD%E5%BF%83%E4%BA%8C%E6%9C%9F',
      'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1541990854961_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E6%B7%B1%E5%9C%B3%E5%9B%BD%E8%B4%B8%E5%A4%A7%E5%8E%A6',
      'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1541990936417_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E6%B7%B1%E5%9C%B3%E7%94%B5%E5%AD%90%E5%A4%A7%E5%8E%A6',
      'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1541990976675_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E6%B7%B1%E5%9C%B3%E4%B8%8A%E6%B5%B7%E5%AE%BE%E9%A6%86'
      ]




for i in range(0,len(wds)):
    try:
        wd=wds[i]
        url=urls[i]
        n=10
        if not os.path.exists(wd):
            os.makedirs(wd)
        
        for i in range(n):
            tem=str(i*60)
            
        #    +tem+'&gsm=0'
            #url='https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1536923171287_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E7%A7%91%E7%BD%97%E6%8B%89%E5%A4%9A%E5%A4%A7%E5%B3%A1%E8%B0%B7'
            url=url+tem+'&gsm=0'
            html=requests.get(url).text
            pic_url=re.findall('"objURL":"(.*?)",',html,re.S)
            n=i*60
            for each in pic_url:
        #        print (each)
                try:
                    pic=requests.get(each,timeout=5)
                except requests.exceptions.ConnectionError:
                    print ('打不开！哼')
                    continue
                
        #        ÓÃÍ¼Æ¬µÄÍøÖ·¸øÍ¼Æ¬ÃüÃû£¬²¢Ö»±£ÁôÍøÖ·µÄ×ÖÄ¸£¬È¥µô·Ç·¨×Ö·ûºÍÊý×Ö
                newS = ''
                for s in each:
                    #isalphaÊÇÖ»±£Áô×ÖÄ¸
                    if s.isalnum(): 
                        newS += s
        #        print ('È¥µô·Ç·¨×Ö·û£º'+newS)
                   
        #        ½ØÈ¡newSÖ±µ½µ¹ÊýµÚÈýÎ»£¬ÒòÎª×îºóÈý¸ö×Ö·û¶¼ÊÇjpgËùÒÔÈ¥µô,Ç°ËÄ¸öÊÇhttp
                ss_str=newS[4:-3]
        #        print(str(n)+"_shemegui_"+ss_str)
                
                str_zero='0000'+str(n)
                image_name=wd.split('\\')[-2]
                image_name=image_name+'_'+str_zero[-5:]
                
                string = wd + image_name + '.jpg'
                print(string)
                
                fp=open(string,'wb')
                fp.write(pic.content)
                fp.close()
                n=n+1
                if n>560:
                    continue
    except:
        continue
        
        
        