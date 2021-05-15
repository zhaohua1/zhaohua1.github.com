import pandas as pd #数据分析、处理
import numpy as np #科学计算包
import matplotlib.pyplot as plt #画图
from sklearn import tree
import pydotplus
from IPython.display import Image,display
import math

#对连续值进行二分类
def data_process(data):
    for k in range(0,4):
        a1=list()
        for i in range(0,150):
            a1.append(data[i][k])
        b1=sorted(a1)
        mid=(b1[74]+b1[75])/2
        for i in range(0,150):
            if(data[i][k]>=mid):
                data[i][k]=1
            else:
                data[i][k]=0
    return data

def ent_d(target):
    a=[0,0,0]
    for i in range(0,len(target)):
        a[target[i]]+=1;
    value=0
    total_a=sum(a)
    for i in range(0,3):
        if a[i]==0:
            value+=0
        else:
            value+=-a[i]/total_a*math.log2(a[i]/total_a)
    return value

def message_value(nature,data,target,shujulaing):
    a=[0,0,0]#属性为0的花数
    b=[0,0,0]#属性nature为1的花数
    for i in range(0,shujulaing):
        if data[i][nature]==0:
            a[target[i]]=a[target[i]]+1
        else:
            b[target[i]]=b[target[i]]+1
    total_a=sum(a);total_b=sum(b);ent_a=0;ent_b=0;bili=total_a/(total_b+total_a)
    #print(a,b)
    for i in range(0,3):#有3种花
        if a[i]==0:
            ent_a+=0
        else:
            ent_a+=-bili*a[i]/total_a*math.log2(a[i]/total_a)
        if b[i]==0:
            ent_b+=0
        else:
            ent_b+=-(1-bili)* b[i] / total_b * math.log2(b[i] / total_b)

    return ent_a+ent_b

def classfier(shuxing,data,target,shujuliang):
    data_0=[];data_1=[];target_0=[];target_1=[]
    for i in range(0,shujuliang):
        if data[i][shuxing]==0:
            data_0.append(data[i])
            target_0.append(target[i])

        else:
            data_1.append(data[i])
            target_1.append(target[i])

    return (data_0,data_1,target_0,target_1)


from sklearn.datasets import load_iris
iris_dataset = load_iris()
data=iris_dataset.data
target=iris_dataset.target
data=data_process(data)
#print(data)
#第一次选择属性3
message_gain=[]
for i in range(0,4):
    message_gain.append(1.585-message_value(i,data,target,150))#原始3*1/3log2(1/3)=1.585
#print(message_gain)
shuxing_13=message_gain.index(max(message_gain))#属性3
(data_30,data_31,target_30,target_31)=classfier(shuxing_13,data,target,150)
#print(target_30,target_31)
#第二层，第二次选择属性1，2
message_gain30=[];message_gain31=[]
for i in range(0,3):
    message_gain30.append(ent_d(target_30)-message_value(i,data_30,target_30,len(data_30)))
    message_gain31.append(ent_d(target_31)-message_value(i,data_31,target_31,len(data_31)))
#print(message_gain30,message_gain31)#分别选择属性（1,2）
(data_30_10,data_30_11,target_30_10,target_30_11)=classfier(1,data_30,target_30,len(data_30))
(data_31_20,data_31_21,target_31_20,target_31_21)=classfier(2,data_31,target_31,len(data_31))
print(target_30_10,target_30_11,target_31_20,target_31_21)
#通过查看标签发现data_30_10,data_30_11,data_31_20基本分清楚，现只对data_31_21再分
#第三次选择不分、不分、不分、属性0
message_gain31_21=[]
ent_31_21=ent_d(target_31_21)
for i in range(0,2):
    message_gain31_21.append(ent_31_21-message_value(i,data_31_21,target_31_21,len(data_31_21)))
print(message_gain31_21)#查看知道属性0信息增益更大
(data_31_21_00,data_31_21_01,target_31_21_00,target_31_21_01)=classfier(0,data_31_21,target_31_21,len(data_31_21))
print(target_31_21_00,target_31_21_01)
#剪枝失败预剪枝
