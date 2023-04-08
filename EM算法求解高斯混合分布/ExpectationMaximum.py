import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# # 定义高斯分布的参数
# mean1, std1 = 164, 3
# mean2, std2 = 176, 5
#
# # 从两个高斯分布中生成各50个样本
# data1 = np.random.normal(mean1, std1, 500)
# data2 = np.random.normal(mean2, std2, 1500)
# data = np.concatenate((data1, data2), axis=0)
#
# # 将数据写入 CSV 文件
# df = pd.DataFrame(data, columns=['height'])
# df.to_csv('height_data.csv', index=False)

# 绘制数据的直方图
# plt.hist(data, bins=20)
# plt.xlabel('Height (cm)')
# plt.ylabel('Count')
# plt.title('Distribution of Heights')
# plt.show()


# 读取数据
df = pd.read_csv('E://python//NLP//height_data.csv')
data = df['height'].values


#定义高斯函数
def f(x,u,sigma):
    value=1/((2*math.pi)**0.5*sigma)*math.e**(-(x-u)**2/(2*sigma**2))
    return value

# 初始化参数
u1 = 160
u2 = 140
sigma1 = 10
sigma2 = 10
q = 0.5

canshu=[]
L=[]
j=300
while j>0:
    if q==1 or q==0:
        break
    miu_boy=[];boy=[];girl=[];miu_girl=[];L_value=0
    for i in range(2000):
        t1=q*f(data[i],u1,sigma1)
        t2=(1-q)*f(data[i],u2,sigma2)
        miu_boy.append(t1/(t1+t2))
        miu_girl.append(t2/(t1+t2))
        boy.append(data[i]*miu_boy[i])
        girl.append(data[i]*miu_girl[i])
        # L_value+=math.log(math.e,t1+t2)
    q=np.mean(miu_boy)


    u1 = sum(boy)/sum(miu_boy)
    sigma1=0
    for i in range(2000):
        sigma1+=miu_boy[i]*(u1-data[i])**2/sum(miu_boy)
    sigma1=sigma1**0.5

    u2=sum(girl)/sum(miu_girl)

    sigma2 = 0
    for i in range(2000):
        sigma2 += miu_girl[i]*(u2-data[i])**2/sum(miu_girl)
    sigma2=sigma2**0.5
    for i in range(2000):
        L_value+=q*f(data[i],u1,sigma1)+(1-q)*f(data[i],u2,sigma2)

    L.append(L_value)
    canshu.append([u1,sigma1,u2,sigma2,q])
    print([u1,sigma1,u2,sigma2,q])
    j-=1
title=['u1','sigma1','u2','sigma2','mixing ratio']
for i in range(5):
    plt.figure(i+1)
    plt.plot([data[i] for data in canshu])
    plt.xlabel('Iteration')
    plt.ylabel('values')
    plt.title(title[i])
    plt.show()
plt.figure(6)
plt.plot(L)
plt.xlabel('Iteration')
plt.ylabel('values')
plt.title('Likelihood function')
plt.show()

