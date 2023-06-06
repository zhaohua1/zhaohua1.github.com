from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

import os
import openai

#下载ClueAI/ChatYuan-large-v1
# from transformers import AutoModel, AutoTokenizer
#
# model_name = "ClueAI/ChatYuan-large-v1"
# save_directory = "./ClueAI-ChatYuan-large-v1"  # 替换为您要保存模型的目录路径
# # 下载模型
# model = AutoModel.from_pretrained(model_name)
# model.save_pretrained(save_directory)
# # 下载分词器
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained(save_directory)

model_directory = "E:/python/NLP/ClueAI-ChatYuan-large-v1"
tokenizer = T5Tokenizer.from_pretrained(model_directory)
model = T5ForConditionalGeneration.from_pretrained(model_directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text
def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")
def answer(text,model,tokenizer, sample=True, top_p=1, temperature=0.7):#sample：是否抽样。生成任务，可以设置为True;top_p：0-1之间，生成的内容越多样
    text = preprocess(text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device)
    if not sample:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, num_beams=1, length_penalty=0.6)
    else:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])
text1 = "“这部电影太好看了，情节紧凑，演员表演出色。”分析一下上述语句的感情色彩"
text2 = "“我感到非常失望，这家餐厅的服务质量太差了。”分析一下上述语句的感情色彩"
text3 =  "“这不可能吧，我很清楚记得电话号码。”分析一下上述语句的感情色彩"
text4 = "“飞机若想尽快实现地面机动转弯，则前轮转弯操纵系统操纵速率应尽量大，而为了保证飞机地面机动不会引起前（主）机轮侧向滑动，又要求前轮转弯操纵速率应限制在一定范围内。”将上述语句翻译成英文"
text5 = "写一首诗歌，关于生活"
text_list=[text1 ,text2 ,text3 ,text4,text5]
for i, input_text in enumerate(text_list):
    input_text = "用户输入：" + input_text + "\n结果："
    print(f"示例{i+1}".center(20, "="))
    output_text = answer(input_text,model,tokenizer)
    print(f"{input_text}{output_text}")
print("结束")


#chatgpt3.5
# 2.获取api-key
openai.api_key = "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"

# 3.使用OpenAI的API完成ChatGPT模型调用
#    model：指的就是ChatGPT模型
#    prompt：向ChatGPT提出的问题
#    max_tokens：返回的最大字符个数
def answer_chatgpt(prompt):
    response = openai.Completion.create(
      model='text-davinci-003',  # 使用ChatGPT-3.5模型,
      prompt=prompt,
      max_tokens=256,
    )
    message=response.choices[0].text
    return message
# 4.打印结果
for i, input_text in enumerate(text_list):
    output_text=answer_chatgpt(input_text)
    input_text = "用户输入：" + input_text + "\n结果："
    print(f"示例{i+1}".center(20, "="))
    print(f"{input_text}{output_text}")
print("结束")

# model_directory = "E:/python/NLP/Langboat"
# tokenizer2 = BloomTokenizerFast.from_pretrained(model_directory)
# model2 = BloomForCausalLM.from_pretrained(model_directory)
# print(tokenizer.batch_decode(model.generate(tokenizer.encode('中国的首都是', return_tensors='pt'))))
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model2.to(device)