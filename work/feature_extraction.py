import numpy as np
import pandas as pd
import re
import os
import time
import jieba
from snownlp import sentiment
from snownlp import SnowNLP
import jieba.posseg as pseg


train_csv_path = r'G:\毕设\数据集\微博\train.csv'
text_csv_path = r'G:\毕设\数据集\微博\text.csv'
train_negative_corpus_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+'/util/negative.txt'
train_positive_corpus_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+'/util/positive.txt'
sentiment_model_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+'/util/sentiment.marshal'
stopwords_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/util/stopwords.txt"

def train_data_read(train_csv_path):
    '''
    训练数据的读入
    df_text  文本信息列
    df_user  用户信息列
    df_image 图片信息列
    '''
    print("正在载入数据中...")
    #微博信息
    df_text = pd.read_csv(train_csv_path,usecols=['id','text','category','label'])   
    #用户信息
    df_user = pd.read_csv(train_csv_path,usecols=['id','userGender','userFollowCount','userFansCount','userWeiboCount','userLocation','userDescription'])
    #微博图片信息
    df_image = pd.read_csv(train_csv_path,usecols=['id','piclist'])
    print("数据载入完成")
    return df_text,df_user,df_image

def text_data_read():
    '''
    文本特征文件的读取
    :return: 文本特征文件
    '''
    df_text = pd.read_csv(text_csv_path)
    return df_text

def text_insert_cols(df_text,new_features_list):
    '''
    增加文本新的特征列，方便后续提取并补充值
    :param df_text: 文本信息
    :return: df_text: 新文本信息dataframe
    '''
    print("正在扩展文本新特征列...")
    col_name = list(df_text.columns)
    col_name = col_name[0:-2]+ new_features_list +col_name[-2:]
    df_text = df_text.reindex(columns=col_name, fill_value=0)
    print("文本新特征列扩展完成...")
    return df_text

def text_feature_extraction(df_text):
    print("开始文本特征提取...")
    # #统计字符串长度
    # df_text['text_length'] = df_text['text'].str.len()
    # #将情感分数列转为float
    # df_text['sentiment_score'] = df_text['sentiment_score'].astype(float)
    # #其余数据统计
    i = 0
    for index, row in df_text.iterrows():
        print("处理进度",i+1,"/",df_text.shape[0])
        #获得需要处理的文本内容
        text_content = row['text']
        # #获得是否含有问号以及问号的数量
        # df_text.at[i,'contains_questmark'], df_text.at[i,'num_questmarks'] = text_questmark(text_content)
        # #获得是否含有感叹号以及感叹号的数量
        # df_text.at[i, 'contains_exclammark'], df_text.at[i, 'num_exclammarks'] = text_exclammark(text_content)
        # #获得是否含有hashtag以及hashtag的数量
        # df_text.at[i, 'contains_hashtag'], df_text.at[i, 'num_hashtags'] = text_hashtag(text_content)
        # #获得是否含有url以及url的数量
        # df_text.at[i, 'contains_URL'], df_text.at[i, 'num_URLs'] = text_url(text_content)
        # #获得是否含有@以及@的数量
        # df_text.at[i, 'contains_mention'], df_text.at[i, 'num_mentions'] = text_mention(text_content)
        # #获得文本情感分数
        # df_text.at[i, 'sentiment_score'] = text_sentiment_score(text_content)
        #词性标注，统计名词、动词、代词数量并返回
        df_text.at[i, 'num_noun'],df_text.at[i, 'num_verb'],df_text.at[i, 'num_pronoun'] = text_part_of_speech(text_content)
        i += 1
    print("文本特征提取结束...")
    return df_text

def text_part_of_speech(text_content):
    '''
    将文本中的汉字进行词性标注并返回数量
    :param text_content: 文本信息
    :return: n名词数量,v动词数量,r代词数量
    '''
    #选取所有的汉字
    words = pseg.cut("".join(re.findall(u"[\u4e00-\u9fa5]",text_content)))
    n = 0 #名词数量
    r = 0 #代词数量
    v = 0 #动词数量
    for w in words:
        if (w.flag.startswith('n')):
            n += 1
        elif (w.flag.startswith('v')):
            v += 1
        elif (w.flag.startswith('r')):
            r += 1
    return n,v,r

def text_questmark(text_content):
    '''
    处理文本中的问号
    :param text_content:处理对象文本
    :return: 是否含有问号（1：有，0：无），问号数量
    '''
    en_questmark_nums = text_content.count("?")
    cn_questmark_nums = text_content.count("？")
    if(en_questmark_nums + cn_questmark_nums > 0):
        return 1,en_questmark_nums + cn_questmark_nums
    else:
        return 0,0

def text_train_sentiment():
    #微博语料训练
    sentiment.train(train_negative_corpus_path,train_positive_corpus_path)
    #保存模型，同时修改snownlp->sentiment->__init__.py->data_path
    sentiment.save(sentiment_model_path)

def text_sentiment_score(text_content):
    '''
    获得文本的情感分数
    0<------------------>1
    消极                积极
    :param text_content: 处理对象文本
    :return: sentiment_score.sentiments 情感分数
    '''
    #去除停用词
    new_text_content = jieba_clear_text(text_content)
    try:
        sentiment_score = SnowNLP(new_text_content).sentiments
    except:
        return 0
    return sentiment_score

def jieba_clear_text(text):
    '''
    jieba分词，并使用自定义停用词表去除停用词以及长度为1的词
    '''
    raw_result = "/".join(jieba.cut(text))
    myword_list = []
    #去除停用词
    for myword in raw_result.split('/'):
        if myword not in stopwords and len(myword.strip())>1:
            myword_list.append(myword)
    return " ".join(myword_list)

def get_stopwords_list():
    '''
    获得停用词的列表
    :return: stopwords：停用词列表
    '''
    my_stopwords = []
    fstop = open(stopwords_path, "r", encoding='UTF-8')
    for eachWord in fstop.readlines():
        my_stopwords.append(eachWord.strip())
    fstop.close()
    return my_stopwords

def text_exclammark(text_content):
    '''
    处理文本中的感叹号
    :param text_content:处理对象文本
    :return: 是否含有感叹（1：有，0：无），感叹数量
    '''
    en_exclammark_nums = text_content.count("!")
    cn_exclammark_nums = text_content.count("！")
    if(en_exclammark_nums + cn_exclammark_nums > 0):
        return 1,en_exclammark_nums + cn_exclammark_nums
    else:
        return 0,0

def text_hashtag(text_content):
    '''
    判断文本中是否存在hashtag
    微博中hashtag由两个#构成，例如 #毕业设计#
    :param text_content: 处理对象文本
    :return: 是否含有hashtag（1：有，0：无），hashtag数量
    '''
    hashtag_nums = text_content.count("#")
    if(hashtag_nums == 0):
        return 0,0
    else:
        return 1,hashtag_nums/2

def text_url(text_content):
    '''
    判断文本中是否存在微博URL
    :param text_content: 处理对象文本
    :return: 是否含有url（1：有，0：无），url数量
    '''
    url = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text_content)
    if url:
        return 1,len(url)
    else:
        return 0,0

def text_mention(text_content):
    '''
    处理微博正文中的用户@
    :param text_content: 处理对象文本
    :return: 是否含有@（1：有，0：无），@数量
    '''
    mention_nums = text_content.count("@")
    if(mention_nums > 0):
        return 1,mention_nums
    else:
        return 0,0


# start = time.time()
# #原始数据的读入
# # df_text,df_user,df_image = train_data_read(train_csv_path)
#
# #微博文本扩展特征数据列
# # new_text_features_list = ['text_length', 'contains_questmark', 'num_questmarks', 'contains_exclammark',
# #                      'num_exclammarks', 'contains_hashtag', 'num_hashtags', 'contains_URL',
# #                      'num_URLs', 'contains_mention', 'num_mentions', 'sentiment_score','num_noun','num_verb','num_pronoun']
# df_text = text_insert_cols(df_text,new_text_features_list)
# #情感分析语料模型训练
# # text_train_sentiment()
# # 读入停用词表
# stopwords = get_stopwords_list()
#
# df_text = text_feature_extraction(df_text)
# df_text.to_csv(text_csv_path,index=0)#不保留行索引
# end = time.time()
# print("运行时间：",end-start)
# # df_user.to_csv(r'G:\毕设\数据集\微博\user.csv',index=0)#不保留行索引
# # df_image.to_csv(r'G:\毕设\数据集\微博\image.csv',index=0)#不保留行索引


start = time.time()
#原始数据的读入
df_text = text_data_read()

#微博文本扩展特征数据列
# new_text_features_list = ['num_noun','num_verb','num_pronoun']
# df_text = text_insert_cols(df_text,new_text_features_list)
#情感分析语料模型训练
# text_train_sentiment()
# 读入停用词表
stopwords = get_stopwords_list()

df_text = text_feature_extraction(df_text)
df_text.to_csv(text_csv_path,index=0)#不保留行索引
end = time.time()
print("运行时间：",end-start)
# df_user.to_csv(r'G:\毕设\数据集\微博\user.csv',index=0)#不保留行索引
# df_image.to_csv(r'G:\毕设\数据集\微博\image.csv',index=0)#不保留行索引