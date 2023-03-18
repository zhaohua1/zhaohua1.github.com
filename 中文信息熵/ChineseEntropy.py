import jieba
import os
import math
import time


import configparser
import requests
from bs4 import BeautifulSoup

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
    wordLen=sum(ChineseCounts.values())
    entropy=0
    for item in ChineseCounts.items():
        word,count=item
        entropy-=(count/wordLen)*math.log((count/wordLen),2)
    return entropy

allText=open(allTextPath,'r',encoding='ANSI').read()#读取是个字符串
entropy=getEntropy(allText,1,stopTxt)
print(entropy)
print('总耗时{:.1f}s'.format(time.time()-stratTime))


