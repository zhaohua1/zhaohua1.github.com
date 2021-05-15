import pandas as pd #数据分析、处理
import numpy as np #科学计算包
import matplotlib.pyplot as plt #画图
from sklearn import tree
from IPython.display import Image,display
import math

#对连续值进行3离散化
def data_process(data):
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
#未分之前信息熵
def ent_d(target):
    a=[0,0,0]
    for i in range(0,len(target)):
        a[target[i]]+=1
    value=0
    total_a=sum(a)
    for i in range(0,3):
        if a[i]==0:
            value+=0
        else:
            value+=-a[i]/total_a*math.log2(a[i]/total_a)
    return value
#分之后信息熵
def message_value(nature,data,target,shujulaing):
    a=[0,0,0]#属性为0的花数
    b=[0,0,0]#属性nature为1的3种花花数
    c=[0,0,0]#nature属性为2的3种花花数
    for i in range(0,shujulaing):
        if data[i][nature]==0:
            a[target[i]]+=1
        elif data[i][nature]==1:
            b[target[i]]+=1
        else:
            c[target[i]]+=1
    total_a=sum(a);total_b=sum(b);total_c=sum(c);ent_a=0;ent_b=0;ent_c=0
    bili0=total_a/shujulaing
    bili1=total_b/shujulaing
    bili2=total_c/shujulaing
    #print(a,b)
    for i in range(0,3):#有3种花
        if a[i]==0:
            ent_a+=0
        else:
            ent_a+=-bili0*a[i]/total_a*math.log2(a[i]/total_a)
        if b[i]==0:
            ent_b+=0
        else:
            ent_b+=-bili1* b[i] / total_b * math.log2(b[i] / total_b)
        if c[i] == 0:
            ent_c += 0
        else:
            ent_c += -bili2 * c[i] / total_c * math.log2(c[i] / total_c)

    return ent_a+ent_b+ent_c

#分类
def classfier(shuxing,data,target,shujuliang):
    data_0=[];data_1=[];data_2=[];target_0=[];target_1=[];target_2=[]
    for i in range(0,shujuliang):
        if data[i][shuxing]==0:
            data_0.append(data[i])
            target_0.append(target[i])

        elif data[i][shuxing]==1:
            data_1.append(data[i])
            target_1.append(target[i])
        else:
            data_2.append(data[i])
            target_2.append(target[i])

    return (data_0,data_1,data_2,target_0,target_1,target_2)


from sklearn.datasets import load_iris
iris_dataset = load_iris()
data=iris_dataset.data
target=iris_dataset.target
data=data_process(data)
#第一次分
message_gain=[]
for i in range(0,4):
    message_gain.append(1.585-message_value(i,data,target,150))#原始3*1/3log2(1/3)=1.585
print(message_gain)#属性3增益最大选择3
shuxing1=message_gain.index(max(message_gain))
(data_30,data_31,data_32,target_30,target_31,target_32)=classfier(shuxing1,data,target,150)
#rint(target_30,target_31,target_32)
#查看标签发现，data_30已经全部分好为种类0
#现在研究data_31,data_32是否继续分
(data_20,data_21,data_22,target_20,target_21,target_22)=classfier(2,data,target,150)
print(target_20,target_21,target_22)

#第二次分之后
message_gain_31=[];message_gain_32=[]
for i in range(0,3):
    message_gain_31.append(ent_d(target_31)-message_value(i,data_31,target_31,len(data_31)))
    message_gain_32.append(ent_d(target_32)-message_value(i,data_32,target_32,len(data_32)))
#print(message_gain_31,message_gain_32)
#查看后分别选择属性2,2
shuxing2_31=message_gain_31.index(max(message_gain_31))
shuxing2_32=message_gain_32.index(max(message_gain_32))
(data_31_20,data_31_21,data_31_22,target_31_20,target_31_21,target_31_22)=classfier(shuxing2_31,data_31,target_31,len(data_31))
(data_32_20,data_32_21,data_32_22,target_32_20,target_32_21,target_32_22)=classfier(shuxing2_32,data_32,target_32,len(data_32))
#print(target_31_20,target_31_21,target_31_22)
#rint(target_32_20,target_32_21,target_32_22)
#查看后31-20,31-22，32-20，为空或分好，32-22已经基本分好
#再对31-22,32-21分类


#第三次分
message_gain_31_22=[];message_gain_32_21=[]
for i in range(0,2):
    message_gain_31_22.append(ent_d(target_31_22)-message_value(i,data_31_22,target_31_22,len(data_31_22)))
    message_gain_32_21.append(ent_d(target_32_21)-message_value(i,data_32_21,target_32_21,len(data_32_21)))
print(message_gain_31_22,message_gain_32_21)
#查看分别选择属性0,1
shuxing3_31_22=message_gain_31_22.index(max(message_gain_31_22))
shuxing3_32_21=message_gain_32_21.index(max(message_gain_32_21))
(data_31_22_00,data_31_22_01,data_31_22_02,target_31_22_00,target_31_22_01,target_31_22_02)=classfier(shuxing3_31_22,data_31_22,target_31_22,len(data_31_22))
(data_32_21_10,data_32_21_11,data_32_21_12,target_32_21_10,target_32_21_11,target_32_21_12)=classfier(shuxing3_32_21,data_32_21,target_32_21,len(data_32_21))
#print(target_31_22_00,target_31_22_01,target_31_22_02)
#print(target_32_21_10,target_32_21_11,target_32_21_12)

