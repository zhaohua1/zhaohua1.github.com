import jieba
import os
import math
import time


import configparser
import requests
from bs4 import BeautifulSoup


import matplotlib.pyplot as plt

#读取txt和url,组合到outfile=allChinese.txt中去
stratTime=time.time()
def read_data(path):
    outfile=path+'\\'+'allChinese.txt'
    files=os.listdir(path)
    with open(outfile,'w',encoding='ANSI')as outF:
        for file in files:
            filePath=os.path.join(path,file)
            with open(filePath,'r',encoding='ANSI') as f:
                if filePath[-4:]=='.txt':
                    text=f.read()
                    outF.write(text)
                else:#'.url'
                    config = configparser.ConfigParser()
                    config.read(filePath)
                    url = config['InternetShortcut']['URL']
                    #print(url)
                    try:
                        # 解析网页内容并提取文本
                        headers = {"user-agent": "Mizilla/5.0"}
                        response = requests.get(url,headers=headers)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        text = soup.get_text()
                        # 将文本添加到文本文件中
                        outF.write(text)
                    except Exception as e:
                        print(f'Error processing {file}: {e}')
            f.close()
    outF.close()
    return files,outfile

stoppath='E:\\课程\\2023春季\\自然语言处理\\DLNLP2023-main\\cn_stopwords.txt'
stopTxt=open(stoppath,'r',encoding='utf-8').read()
path='E:\\课程\\2023春季\\自然语言处理\\jyxstxtqj_downcc.com'
files,allTextPath=read_data(path)

def countleader(ChineseCounts,n):
    dictCountLeader={}  # 计算以前n-1个词为首的n元组的总个数
    for item in ChineseCounts.items():
        word, count = item
        if n>=2:
           dictCountLeader[word[0:n-1]]=dictCountLeader.get(word[0:n-1],0)+count
        else:
            dictCountLeader=ChineseCounts
    return dictCountLeader
#计算n-gram信息熵

def getEntropy(text,n,stopTxt):
    #将text划分为n-gram序列
    ChineseCounts={}
    words=[w for w in (jieba.lcut(allText)) if w not in stopTxt]
    wordsPairs=[(words[i:i+n])for i in range(len(words)-n+1)]
    for pair in wordsPairs:
        pair=tuple(pair)
        ChineseCounts[pair]=ChineseCounts.get(pair,0)+1
    #计算熵
    #获取n元组总个数
    wordLen=sum(ChineseCounts.values())
    # 获取前n-1个词为首的n元组的总个数

    dictCountLeader = countleader(ChineseCounts, n)
    #如果是1元模型,则dictCountLeader=ChineseCounts
    entropy=0
    for item in ChineseCounts.items():
        word,count=item
        # 联合概率每个n元词组在语料库中出现的频率
        probilityUnion = count / wordLen
        # 获取条件概率可近似等于每个n元词组在
        # 语料库中出现的频率与以该n元词组的前n-1词为词首的二元词组的频数的比值。
        probilityCondition=probilityUnion
        if n>=2:
            probilityCondition = count/dictCountLeader[word[0:n-1]]
        entropy-=probilityUnion*math.log(probilityCondition,2)
    return entropy

allText=open(allTextPath,'r',encoding='ANSI').read()#读取成为字符串
entropy=getEntropy(allText,3,stopTxt)
print(entropy)
print('总耗时{:.1f}s'.format(time.time()-stratTime))


# entropy=[]
# for i in range(1,6):
#     entropyValue=getEntropy(allText,i,stopTxt)
#     entropy.append(i)
#
# print(entropy)
# print('总耗时{:.1f}s'.format(time.time()-stratTime))
# X=[]
# for i in range(5):
#     X.append(str(i+1)+'-gram')
# plt.bar(X,entropy)
# plt.xlabel('n-gram')
# plt.ylabel('entropy value')
# plt.show()
