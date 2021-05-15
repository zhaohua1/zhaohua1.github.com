
#采用3分法，再前者已经得到只用属性3分类得到决策树，和只用属性2分类得到决策树基础上，使用两者的均值得到新的决策树
#使用属性3值为（0~2）分别分类成0,1,2，使用属性2值为（0~2）分别分类成0,1,2类花
import pandas as pd #数据分析、处理
import numpy as np #科学计算包
import matplotlib.pyplot as plt #画图
from sklearn import tree
from IPython.display import Image,display
import math
#对连续值进行3离散化
def data_process3(data):
    for k in range(0,4):
        a1=list()
        for i in range(0,150):
            a1.append(data[i][k])
        b1=sorted(a1)

        for i in range(0,150):
            if(data[i][k]>=b1[99]):
                data[i][k]=2
            elif(data[i][k]>=b1[49]):
                data[i][k]=1
            else:
                data[i][k]=0
    return data


def classfier(data,target):
    label=0;label_3=0;label_2=0;ture_test=0
    for i in range(0,150):
        label_3=data[i][3]
        label_2=data[i][2]
        label=round((label_2+label_3)/2)
        #print(label_2,label_3)
        if label==target[i]:
            ture_test+=1
    return ture_test/150

from sklearn.datasets import load_iris
iris_dataset = load_iris()
data=iris_dataset.data
target=iris_dataset.target
data=data_process3(data)
print('分类准确率：{:.2f}%'.format(classfier(data,target)*100))
