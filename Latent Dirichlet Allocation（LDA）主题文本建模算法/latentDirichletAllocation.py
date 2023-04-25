import os
import jieba
import numpy as np
import random
from sklearn.model_selection import train_test_split
from gensim import corpora, models
import wordcloud

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt

def readFiles(path,stopPath,n_gram):
    with open(stopPath,'r', encoding='utf-8') as f:
        stopTxt=[line.strip() for line in f.readlines()]
        stopTxt.append('')
    files=os.listdir(path)
    txtDict={}
    # 去掉停词
    ad = ['本书来自www.cr173.com免费txt小说下载站','更多更新免费电子书请关注www.cr173.com',
          '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....',
          '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']

    for file in files:
        pathfile=os.path.join(path,file)
        with open(pathfile, 'r', encoding='ANSI') as f:
            if pathfile[-4:] == '.txt':
                text = f.read()
                for i in ad:
                    text=text.replace(i,'')
                split_text=[word for word in jieba.lcut(text)]
                #去掉停词
                split_text = [word for word in split_text if word not in stopTxt]
                new_split_text=[]
                for i in range(len(split_text) -n_gram+ 1):
                    word = ''
                    for j in range(n_gram):
                        word += split_text[i + j]
                        new_split_text.append(word)
                txtDict[file.split('.')[0]]=new_split_text
                # print(file.split('.')[0],'总词数',len(split_text))
    return txtDict
# stopPath='E:\\课程\\2023春季\\自然语言处理\\DLNLP2023-main\\cn_stopwords.txt'
# path='E:\\课程\\2023春季\\自然语言处理\\jyxstxtqj_downcc.com'
# txt_dict=readFiles(path,stopPath)


class lda_train():
    def __init__(self,txt_dict,paragraph_num=15, paragraph_length=500, topics_num=10):
        self.txt_dict=txt_dict
        self.paragraph_num=paragraph_num
        self. paragraph_length= paragraph_length
        self.topics_num=topics_num
    def preprocess(self):
        txt_list=[]
        txt_labels=[]
        # 随机划分段落
        for label,txt in self.txt_dict.items():
            for i in range(self.paragraph_num):
                random.seed(0)
                random_int = random.randint(0, self.paragraph_length)
                step=len(txt)//self.paragraph_num
                txt_list.append(txt[random_int+i*step:random_int +(i+1)*step])
                txt_labels.append(label)
        #处理标签成整数
        for i in range(len(txt_labels)):
            txt_labels[i]=i//self.paragraph_num
        #生成训练集和测试集
        train_txt, test_txt, train_labels, test_labels = \
            train_test_split(txt_list, txt_labels, test_size=0.9, random_state=1)
        return txt_list,txt_labels,train_txt, test_txt, train_labels, test_labels

    def train_lda(self):
        txt_list,txt_labels,train_txt, test_txt, train_labels, test_labels=self.preprocess()
        dictionary=corpora.Dictionary(txt_list)
        corpus_train=[dictionary.doc2bow(doc) for doc in train_txt]
        corpus_test=[dictionary.doc2bow(document=doc) for doc in test_txt]

        lda = models.LdaModel(corpus=corpus_train, id2word=dictionary, num_topics=self.topics_num)
        # 获取训练集和测试集的每个段落的主题分布
        topics_train = lda.get_document_topics(corpus_train, minimum_probability=0)
        topics_test = lda.get_document_topics(corpus_test, minimum_probability=0)
        feature_train = []
        feature_test = []

        for i in range(0, len(topics_train)):
            feature_train.append([k[1] for k in topics_train[i]])

        for i in range(0, len(topics_test)):
            feature_test.append([k[1] for k in topics_test[i]])

        print('训练集特征矩阵大小为:', np.array(feature_train).shape)
        print('测试集特征矩阵大小为:', np.array(feature_test).shape)

        clf = MultinomialNB()
        clf.fit(feature_train,train_labels)
        predict_train = clf.predict(feature_train)
        predict_test = clf.predict(feature_test)
        accuracy_train = clf.score(feature_train , train_labels)
        print(f'Train Accuracy: {100 * accuracy_train:.2f}%')
        accuracy_test = clf.score(feature_test, test_labels)
        print(f'Test Accuracy: {100 * accuracy_test:.2f}%')
        return accuracy_train, accuracy_test



stopPath='E:\\课程\\2023春季\\自然语言处理\\DLNLP2023-main\\cn_stopwords.txt'
path='E:\\课程\\2023春季\\自然语言处理\\jyxstxtqj_downcc.com'
txt_dict=readFiles(path,stopPath,n_gram=1)

# 制作词云
# ls=[w for w in (jieba.lcut(allText)) if w not in stopTxt]
# txt=[]
# for i in txt_dict.values():
#     for j in i:
#         txt.append(j)
# txt=" ".join(txt)
# from imageio.v3 import imread
# mask=imread("E:\\python\\NLP\\img.png")
# w=wordcloud.WordCloud(font_path="msyh.ttc",width=1000,height=700,
#                       background_color="white",max_words=150,mask=mask)
# w.generate(txt)
# w.to_file('E:\\python\\NLP\\result.png')



acc_train=[]
acc_test=[]
topics_num=[5,10,15,20,50,100,200,300]
# topics_num=[10]
n_gram=[1,2,3,4]
for i_gram in n_gram:
    txt_dict=readFiles(path,stopPath,n_gram=i_gram)
    LDA = lda_train(txt_dict=txt_dict, paragraph_num=50,
                    paragraph_length=1000, topics_num=50)
    accuracy_train, accuracy_test = LDA.train_lda()
    acc_train.append(accuracy_train)
    acc_test.append(accuracy_test)
X_gram=[str(i)+'-gram' for i in n_gram]
plt.figure(1)
plt.bar(X_gram,acc_train)
plt.show()
plt.figure(2)
plt.bar(X_gram,acc_test)
plt.show()


#     acc_test.append(accuracy_test)
# for topic_num in topics_num:
#     LDA=lda_train(txt_dict=txt_dict,paragraph_num=50,
#               paragraph_length=1000, topics_num=topic_num)
#     accuracy_train, accuracy_test=LDA.train_lda()
#     acc_train.append(accuracy_train)
#     acc_test.append(accuracy_test)
#
# topics=[]
# for i in topics_num:
#     topics.append('topics: '+str(i))
# plt.figure(1)
# plt.bar(topics,acc_train)
# plt.show()
# plt.figure(2)
# plt.bar(topics,acc_test)
# plt.show()

