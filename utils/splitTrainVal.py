import os
import random
'''
Attention:
    After run this code,we need to delete the last '\n' in train.txt or val.txt manually
'''
dataDir='../dataset/datalist.txt'
trainDir='../dataset/train.txt'
valDir='../dataset/val.txt'
train=open(trainDir,'w')
val=open(valDir,'w')
with open(os.path.join(dataDir), 'r') as fd:
    imgs = fd.readlines()
#print(imgs)
random.shuffle(imgs)
ratio = 0.9
nums = len(imgs)
#print(imgs[:int(nums*ratio)])
val.writelines(imgs[int(nums*ratio):])
train.writelines(imgs[:int(nums*0.9)])
val.close()
train.close()
#类别数统计
from collections import Counter
with open(os.path.join(dataDir), 'r') as fd:
    imgs = fd.readlines()
labels=[int(per.strip('\n').split(',')[-1].strip(' ')) for per in imgs]
c = dict(Counter(labels))
cls=[0]*len(c)
for key in c.keys():
    cls[key]=c[key]
print(cls,'\n',len(cls))


