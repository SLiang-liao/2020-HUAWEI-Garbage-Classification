import os
import cv2
import glob
import random
img_dir="/home/slleo/桌面/Huawei-cloud-垃圾分类/dataset/train_data"
names=os.listdir(img_dir)
label_txt='../dataset/datalist.txt'
w=open(label_txt,'w')
label_str=""
for name in names:
    if name[-4:]=='.txt':
        with open(os.path.join(img_dir,name),'r') as f: 
            tmp=f.readline()
            if len(tmp)>0:
                tmp = tmp.strip('\n')+'\n'
                print(tmp)
            #input()
        label_str+=tmp
label_str=label_str.strip('\n')
w.write(label_str)
w.close()
