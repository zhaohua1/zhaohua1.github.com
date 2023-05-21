import jieba
import os
import torch
import torch.nn as nn
from gensim.models import Word2Vec


from tqdm import tqdm
import random
import re
import time
import numpy as np

def preprocess_text(novels_dir):
    # 去掉无意义的词
    ad = ['本书来自www.cr173.com免费txt小说下载站', '更多更新免费电子书请关注www.cr173.com',
          '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '『', '』', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b','，','=']

    # 读取文本数据并进行预处理
    novel_dataset = []
    for root, _, files in os.walk(novels_dir):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='ANSI') as f:
                text = f.read()
                for i in ad:
                    text=text.replace(i,'')
                text=[word for word in jieba.lcut(text)]
                novel_dataset.append(text)
    return novel_dataset


path='E:\\课程\\2023春季\\自然语言处理\\jyxstxtqj_downcc.com'
novel_dataset=preprocess_text(path)

#将词转化为词向量
#转化为word2vec词向量
# def w2v(novel_dataset):
#     # 将分词结果列表作为输入
#     corpus = novel_dataset
#
#     # 训练Word2Vec模型
#     w2v_model = Word2Vec(corpus,vector_size=1,window=5, min_count=5, workers=4,sg=0)#默认size=100
#
#     # 获取词汇表和词向量
#     vocab = w2v_model.wv.index_to_key
#     word_vectors = w2v_model.wv.vectors
#     #w2v_model.wv['金庸']
#     # 根据词汇表创建词向量字典
#     word2vec = dict(zip(vocab, word_vectors))
#     return w2v_model,word2vec
# def convert_text_to_vectors(word, word2vec):
#     vectors = []
#     for word in text:
#         if word in word2vec.keys():
#             vectors.append(word2vec[word])
#     return torch.tensor(vectors)
# w2v_model,word2vec=w2v(novel_dataset=novel_dataset)
# novel_dataset_all=[]
# for i in range(len(novel_dataset)):
#     novel_dataset_all.extend(novel_dataset[i])
# print(len(novel_dataset_all))
# noval_dataset_all_vector=[word2vec[word] for word in novel_dataset_all if word in word2vec.keys()]
# # print(noval_dataset_all_vector[0].size())

# print(novel_dataset[0])
print(len(novel_dataset[0]))#第一本书的长度
novel_dataset_all=[]
# for i in range(len(novel_dataset)):
#     novel_dataset_all.extend(novel_dataset[i])
novel_dataset_all.extend(novel_dataset[0])
# novel_dataset_all.extend(novel_dataset[1])
print(len(novel_dataset_all))


# 构建字典映射每个中文字到索引的关系
word2index = {}

for word in novel_dataset_all:
    if word not in word2index:
        word2index[word] = len(word2index)

index2word = {index: word for word, index in word2index.items()}

# 将中文转换为索引
novels_index_lst = [word2index[word] for word in novel_dataset_all]


class LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        outputs = self.fc_out(outputs)

        return outputs

vocab_size = len(word2index) # 词典中总共的词数，是文章有多少个不同的词
embed_size = 30 # 每个词语嵌入特征数
hidden_size = 512 # LSTM的每个时间步的每一层的神经元数量
num_layers = 2 # LSTM的每个时间步的隐藏层层数

max_epoch = 10
batch_size = 16
learning_rate = 0.001
sentence_len = 20
train_lst = [i for i in range(0,10000)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = LSTM(vocab_size, embed_size, hidden_size, num_layers, vocab_size).to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(max_epoch)):# 打印循环中的进度条
    for i in train_lst:
        inputs = torch.tensor([novels_index_lst[j:j + sentence_len] for j in range(i, i + batch_size)]).to(device)
        targets = torch.tensor([novels_index_lst[j + 1:j + 1 + sentence_len] for j in range(i, i + batch_size)]).to(
            device)
        outputs = model(inputs)

        loss = criterion(outputs.view(outputs.size(0) * outputs.size(1), -1), targets.view(-1))
        model.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()

'''保存模型'''
save_path = 'lstm.pth'
torch.save(model, save_path)


def tensor_to_str(index2word_dict, class_tensor):
    # 将张量转换为字符串
    class_lst = list(class_tensor)
    words = [index2word_dict[int(index)] for index in class_lst]

    # 将列表中的词语连接为一个字符串
    sentence = ''.join(words)
    return sentence


test_model = torch.load('lstm.pth').to('cpu')

generate_length = 30  # 测试长度
test_set = [novels_index_lst[i:i + sentence_len] for i in range(10000, 30000, 2000)]
target_set = [novels_index_lst[i:i + sentence_len + generate_length] for i in range(10000, 30000, 2000)]

for i in range(0, len(test_set)):
    generate_lst = []
    generate_lst.extend(test_set[i])
    for j in range(0, generate_length):
        inputs = torch.tensor(generate_lst[-sentence_len:])  # 选取生成词语列表的最后sentence_len个元素作为下一次模型的输入
        outputs = test_model(inputs)

        predicted_class = torch.argmax(outputs, dim=-1)

        generate_lst.append(int(predicted_class[-1]))

    input_sentence = tensor_to_str(index2word, test_set[i])
    generate_sentence = tensor_to_str(index2word, generate_lst)
    target_sentence = tensor_to_str(index2word, target_set[i])
    # 打印生成结果
    print('测试结果', i)
    print('初始输入句：\n', input_sentence)
    print('模型生成句：\n', generate_sentence)
    print('期待生成句：\n', target_sentence)
    print('---' * 20)