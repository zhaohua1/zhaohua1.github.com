import random

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


def judge(data_train,target_train,data_test,target_test):#train 0-30,30-60,60-90分别为花0,1,2
    p0=[];p1=[];p2=[];a=[0,0,0]#每种花中12维概率
    for i in range(0,4):#由于有4个属性，每个属性有3个离散值，共12维
        for j in range(0,30):
            if data_train[j][i]==0:
                a[0]+=1
            elif data_train[j][i]==1:
                a[1]+=1
            else:
                a[2]+=1
        p0.append(a)
        a=[0,0,0]
    for i in range(0,4):#由于有4个属性，每个属性有3个离散值，共12维
        for j in range(30,60):
            if data_train[j][i]==0:
                a[0]+=1
            elif data_train[j][i]==1:
                a[1]+=1
            else:
                a[2]+=1
        p1.append(a)
        a = [0, 0, 0]
    for i in range(0, 4):  # 由于有4个属性，每个属性有3个离散值，共12维
        for j in range(60, 90):
            if data_train[j][i] == 0:
                a[0] += 1
            elif data_train[j][i] == 1:
                a[1] += 1
            else:
                a[2] += 1
        p2.append(a)
        a=[0,0,0]
    #测试集
    ture_test=0
    for i in range(0,60):
        label=0;P0=1;P1=1;P2=1
        for j in range(0,4):
            a = data_test[i][j]
            a=int(a)
            if p0[j][a]==0:
                P0*=1/31
            else:
                P0*=p0[j][a]/31
        for j in range(0, 4):
            a = data_test[i][j]
            a = int(a)
            if p1[j][a]==0:
                P1*=1/31
            else:
                P1 *= p1[j][a] / 31
        for j in range(0, 4):
            a = data_test[i][j]
            a = int(a)
            if p2[j][a]==0:
                P2*=1/31
            else:
                P2*=p2[j][a] / 31
        M=max(P0,P1,P2)
        if P0==M:
            label=0
        if P1==M:
            label=1
        else:
            label=2
        if label==target_test[i]:
            ture_test+=1
        print('{:.3f} {:.3f} {:.3f} {}'.format(P0,P1,P2,target_test[i]))
    return ture_test/60

from sklearn.datasets import load_iris
iris_dataset = load_iris()
data=iris_dataset.data
target=iris_dataset.target
data=data_process(data)
#训练集，测试集
data_train=[];data_test=[];target_train=[];target_test=[]
for i in range(0,30):
    data_train.append(data[i])
    target_train.append(target[i])
for i in range(50, 80):
    data_train.append(data[i])
    target_train.append(target[i])
for i in range(100,130):
    data_train.append(data[i])
    target_train.append(target[i])
for i in range(30,50):
    data_test.append(data[i])
    target_test.append(target[i])
for i in range(80,100):
    data_test.append(data[i])
    target_test.append(target[i])
for i in range(130,150):
    data_test.append(data[i])
    target_test.append(target[i])

print('朴素贝叶斯对测试集准确率：{}%'.format(judge(data_train,target_train,data_test,target_test)*100))
