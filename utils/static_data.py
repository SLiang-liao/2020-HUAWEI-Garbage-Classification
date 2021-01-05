import os
import random
import matplotlib.pyplot as plt
'''
Attention:
    After run this code,we need to delete the last '\n' in train.txt or val.txt manually
'''
dataDir='../dataset/datalist.txt'
trainDir='../dataset/train.txt'
valDir='../dataset/val.txt'
testDir='../dataset/test.txt'
data = [dataDir,trainDir,valDir,testDir]
#类别数统计
from collections import Counter
for i,each in enumerate(data):
    x = [i for i in range(43)]
    print(each)
    with open(os.path.join(each), 'r') as fd:
        imgs = fd.readlines()
    labels=[int(per.strip('\n').split(',')[-1].strip(' ')) for per in imgs]
    c = dict(Counter(labels))
    cls=[0]*len(c)
    for key in c.keys():
        cls[key]=c[key]
    print(cls,'\n',len(cls))
    x = [i for i in range(len(cls))]
    plt.subplot(411+i)
    
    plt.bar(x,cls)
    plt.title(each)
    plt.xlabel('class_nums')
    plt.ylabel('nums')
plt.show()
