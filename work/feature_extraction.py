import gensim
import numpy as np
import pandas as pd
import re
import os
import time
import jieba
import cv2
import json
import urllib
import random
import hashlib
from snownlp import sentiment
from snownlp import SnowNLP
import jieba.posseg as pseg
from gensim.models import word2vec
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
from collections import Counter
from scipy.linalg import norm


train_csv_path = r'G:\毕设\数据集\微博\train.csv'
text_csv_path = r'G:\毕设\数据集\微博\text.csv'
user_csv_path = r'G:\毕设\数据集\微博\user.csv'
image_csv_path = r'G:\毕设\数据集\微博\image.csv'
en_imagenet_class_path = r'G:\毕设\数据集\微博\imagenet_class_index.json'
cn_imagenet_class_path = r'G:\毕设\数据集\微博\imagenet_class_cn.json'
image_class_vgg19_score_path = r'G:\毕设\数据集\微博\image_class_vgg19.txt'
image_class_resnet50_score_path = r'G:\毕设\数据集\微博\image_class_resnet50.txt'
train_negative_corpus_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+'/util/negative.txt'
train_positive_corpus_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+'/util/positive.txt'
sentiment_model_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+'/util/sentiment.marshal'
stopwords_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/util/stopwords.txt"
word2vec_txt_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/util/word2vec_corpus.txt"
word2vec_model_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/util/text8.model"
possentiwords_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/util/possentiwords.txt"
negsentiwords_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/util/negsentiwords.txt"
appid = '20190716000318328'
secretKey = '7pjdBCkaUodI5eNqsBWB'
url_baidu = 'http://api.fanyi.baidu.com/api/trans/vip/translate'

def train_data_read(train_csv_path):
    """
    训练数据的读入
    df_text  文本信息列
    df_user  用户信息列
    df_image 图片信息列
    """
    logging.info("正在载入数据中...")
    #微博信息
    df_text = pd.read_csv(train_csv_path,usecols=['id','text','category','label'])   
    #用户信息
    df_user = pd.read_csv(train_csv_path,usecols=['id','userGender','userFollowCount','userFansCount','userWeiboCount','userLocation','userDescription'])
    #微博图片信息
    df_image = pd.read_csv(train_csv_path,usecols=['id','piclist'])
    logging.info("数据载入完成")
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
    logging.info("正在扩展文本新特征列...")
    col_name = list(df_text.columns)
    # 插入新列之前列名去重
    col_name = col_name + sorted(set(new_features_list) - set(col_name), key=new_features_list.index)
    df_text = df_text.reindex(columns=col_name, fill_value=0)
    logging.info("文本新特征列扩展完成")
    return df_text


def text_feature_extraction(df_text):
    logging.info("开始文本特征提取...")
    # #统计字符串长度
    # df_text['text_length'] = df_text['text'].str.len()
    # #将情感分数列转为float
    # df_text['sentiment_score'] = df_text['sentiment_score'].astype(float)
    # for j in range(1,101):
    #     df_text['word2vec_'+str(j)] = df_text['word2vec_'+str(j)].astype(float)
    # #其余数据统计
    i = 0
    for index, row in df_text.iterrows():
        logging.info("处理进度"+str(i+1)+"/"+str(df_text.shape[0]))
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
        # #词性标注，统计名词、动词、代词数量并返回
        # df_text.at[i, 'num_noun'],df_text.at[i, 'num_verb'],df_text.at[i, 'num_pronoun'] = text_part_of_speech(text_content)
        # #计算每条微博正文的词向量均值
        df_text.at[i,-107:-7] = text_compute_word2vec(text_content).tolist()
        # #获得每条微博的积极词汇数、消极词汇数
        # df_text.at[i, 'num_possentiwords'], df_text.at[i, 'num_negsentiwords'] = text_pos_neg_sentiwords(text_content)
        #获取新闻是否含有第一人称、第二人称、第三人称
        # df_text.at[i, 'contains_firstorderpron'], df_text.at[i, 'contains_secondorderpron'], df_text.at[i, 'contains_thirdorderpron'] = text_get_fir_sec_thi_orderpron(text_content)
        i += 1
    logging.info("文本特征提取结束...")
    return df_text

def text_get_fir_sec_thi_orderpron(text_content):
    """
    统计第一、二、三人称是否存在于微博中
    :param text_content:
    :return: has_first, has_second, has_third（0:不包含,1:包含)
    """
    has_first = 0 #第一人称
    has_second = 0 #第二人称
    has_third = 0 #第三人称
    if text_content.find('我') != -1:
        has_first = 1
    elif text_content.find('你') != -1:
        has_second = 1
    elif text_content.find('他') != -1 or text_content.find('她') != -1 or text_content.find('它') != -1:
        has_third = 1
    return has_first, has_second, has_third


def text_pos_neg_sentiwords(text_content):
    # 去除停用词的分词String
    new_text_content = jieba_clear_text(text_content)
    #将词组转成list
    list_new_text_content = new_text_content.split(' ')
    #统计积极词、消极词
    num_pos = 0
    num_neg = 0
    for word in list_new_text_content:
        if word in possentiwords:
            num_pos += 1
        elif word in negsentiwords:
            num_neg += 1
    return num_pos,num_neg



def text_part_of_speech(text_content):
    """
    将文本中的汉字进行词性标注并返回数量
    :param text_content: 文本信息
    :return: n名词数量,v动词数量,r代词数量
    """
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
    """
    处理文本中的问号
    :param text_content:处理对象文本
    :return: 是否含有问号（1：有，0：无），问号数量
    """
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
    """
    获得文本的情感分数
    0<------------------>1
    消极                积极
    :param text_content: 处理对象文本
    :return: sentiment_score.sentiments 情感分数
    """
    #去除停用词
    new_text_content = jieba_clear_text(text_content)
    try:
        sentiment_score = SnowNLP(new_text_content).sentiments
    except:
        return 0
    return sentiment_score

def jieba_clear_text(text):
    """
    jieba分词，并使用自定义停用词表去除停用词以及长度为1的词
    """
    raw_result = "$".join(jieba.cut(text))
    myword_list = []
    #去除停用词
    for myword in raw_result.split('$'):
        if myword not in stopwords:
            myword_list.append(myword)
    return " ".join(myword_list)

def get_stopwords_list():
    """
    获得停用词的列表
    :return: stopwords：停用词列表
    """
    my_stopwords = []
    fstop = open(stopwords_path, "r", encoding='UTF-8')
    for eachWord in fstop.readlines():
        my_stopwords.append(eachWord.strip())
    fstop.close()
    return my_stopwords

def get_possentiwords_list():
    """
    获得积极词汇列表
    :return:
    """
    my_possentiwords = []
    fp = open(possentiwords_path, "r", encoding='UTF-8')
    for eachWord in fp.readlines():
        my_possentiwords.append(eachWord.strip())
    fp.close()
    return my_possentiwords

def get_negsentiwords_list():
    """
    获得消极词汇列表
    :return:
    """
    my_negsentiwords = []
    fn = open(negsentiwords_path, "r", encoding='UTF-8')
    for eachWord in fn.readlines():
        my_negsentiwords.append(eachWord.strip())
    fn.close()
    return my_negsentiwords

def text_exclammark(text_content):
    """
    处理文本中的感叹号
    :param text_content:处理对象文本
    :return: 是否含有感叹（1：有，0：无），感叹数量
    """
    en_exclammark_nums = text_content.count("!")
    cn_exclammark_nums = text_content.count("！")
    if(en_exclammark_nums + cn_exclammark_nums > 0):
        return 1,en_exclammark_nums + cn_exclammark_nums
    else:
        return 0,0

def text_hashtag(text_content):
    """
    判断文本中是否存在hashtag
    微博中hashtag由两个#构成，例如 #毕业设计#
    :param text_content: 处理对象文本
    :return: 是否含有hashtag（1：有，0：无），hashtag数量
    """
    hashtag_nums = text_content.count("#")
    if(hashtag_nums == 0):
        return 0,0
    else:
        return 1,hashtag_nums/2

def text_url(text_content):
    """
    判断文本中是否存在微博URL
    :param text_content: 处理对象文本
    :return: 是否含有url（1：有，0：无），url数量
    """
    url = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text_content)
    if url:
        return 1,len(url)
    else:
        return 0,0

def text_mention(text_content):
    """
    处理微博正文中的用户@
    :param text_content: 处理对象文本
    :return: 是否含有@（1：有，0：无），@数量
    """
    mention_nums = text_content.count("@")
    if(mention_nums > 0):
        return 1,mention_nums
    else:
        return 0,0

def text_train_word2vec_model(word2vec_txt_path,word2vec_model_path):
    """
    训练word2vec词向量模型
    :param word2vec_txt_path: 语料路径
    :param word2vec_model_path: 模型保存路径
    :return: 词向量模型
    """
    sentences = word2vec.Text8Corpus(word2vec_txt_path)
    model = word2vec.Word2Vec(sentences,size=100,workers=4)
    # 1.sentences：可以是一个List，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
    # 2.sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
    # 3.size：是指输出的词的向量维数，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    # 4.window：为训练的窗口大小，8表示每个词考虑前8个词与后8个词（实际代码中还有一个随机选窗口的过程，窗口大小<=5)，默认值为5。
    # 5.alpha: 是学习速率
    # 6.seed：用于随机数发生器。与初始化词向量有关。
    # 7.min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。
    # 8.max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
    # 9.sample: 表示 采样的阈值，如果一个词在训练样本中出现的频率越大，那么就越会被采样。默认为1e-3，范围是(0,1e-5)
    # 10.workers:参数控制训练的并行数。
    # 11.hs: 是否使用HS方法，0表示: Negative Sampling，1表示：Hierarchical Softmax 。默认为0
    # 12.negative: 如果>0,则会采用negative samping，用于设置多少个noise words
    # 13.cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（default）则采用均值。只有使用CBOW的时候才起作用。
    # 14.hashfxn： hash函数来初始化权重。默认使用python的hash函数
    # 15.iter： 迭代次数，默认为5。
    # 16.trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）或者一个接受()并返回RU·E_DISCARD,uti·s.RU·E_KEEP或者uti·s.RU·E_DEFAU·T的函数。
    # 17.sorted_vocab： 如果为1（defau·t），则在分配word index 的时候会先对单词基于频率降序排序。
    # 18.batch_words：每一批的传递给线程的单词的数量，默认为10000

    model.save(word2vec_model_path)
    return model

def text_load_word2vec_model(word2vec_model_path):
    """
    加载训练完成的word2vec词向量模型
    :param word2vec_model_path: 模型路径
    :return: 词向量模型
    """
    model = word2vec.Word2Vec.load(word2vec_model_path)
    return model

def text_get_clear_word2vec_corpus(word2vec_txt_path):
    """
    从原始微博文本获得word2vec语料文本
    :param word2vec_txt_path: 语料保存位置
    :return: 0
    """
    with open(word2vec_txt_path, 'a') as f:
        for index, row in df_text.iterrows():
            text_content = row['text']
            raw_txt = jieba_clear_text("".join(re.findall(u"[\u4e00-\u9fa5]", text_content)))
            f.write(raw_txt + "\n")
    logging.info("清理word2vec语料文本结束")

def text_compute_word2vec(text_content):
    raw_txt_list = jieba_clear_text("".join(re.findall(u"[\u4e00-\u9fa5]", text_content))).split(' ')
    text_word2vec_score_list = []
    for word in raw_txt_list:
        try:
            #自己训练的词库用这一句
            text_word2vec_score_list.append(model_word2vec.wv[word])
            # text_word2vec_score_list.append(model_word2vec[word])
        except KeyError:
            text_word2vec_score_list.append(np.zeros(100))
    result_mean_array = np.mean(np.array(text_word2vec_score_list),axis=0)
    return result_mean_array


def user_data_read():
    """
    用户特征文件的读取
    :return: 用户特征文件
    """
    df_user = pd.read_csv(user_csv_path)
    return df_user

def user_insert_cols(df_user,new_features_list):
    """
    增加用户新的特征列，方便后续提取并补充值
    :param df_user: 用户信息
    :return: df_user: 新用户信息dataframe
    """
    logging.info("正在扩展用户新特征列...")
    col_name = list(df_user.columns)
    col_name = col_name + new_features_list
    df_user = df_user.reindex(columns=col_name, fill_value=0)
    logging.info("用户新特征列扩展完成")
    return df_user

def user_feature_extraction(df_user):
    logging.info("开始用户特征提取...")
    #将 关注/粉丝比 列转为float
    df_user['folfans_ratio'] = df_user['folfans_ratio'].astype(float)
    #其余数据统计
    i = 0
    for index, row in df_user.iterrows():
        logging.info("处理进度"+str(i+1)+"/"+str(df_user.shape[0]))
        #获得需要处理的文本内容
        user_follow_count = row['user_follow_count']
        user_fans_count = row['user_fans_count']
        #计算 关注/粉丝比
        df_user.at[i,'folfans_ratio'] = user_compute_folfans_ratio(user_follow_count,user_fans_count)
        i += 1
    logging.info("用户特征提取结束...")
    return df_user

def user_compute_folfans_ratio(user_follow_count,user_fans_count):
    """
    计算关注/粉丝比
    :param user_follow_count: 关注数
    :param user_fans_count: 粉丝数
    :return:
    """
    if( user_fans_count == 0):
        return 0
    else:
        return user_follow_count/user_fans_count

def image_data_read():
    """
    图片特征文件的读取
    :return: 图片特征文件
    """
    df_image = pd.read_csv(image_csv_path)
    return df_image

def image_insert_cols(df_image,new_features_list):
    """
    增加图片新的特征列，方便后续提取并补充值
    :param df_image: 图片信息
    :return: df_image: 新图片信息dataframe
    """
    logging.info("正在扩展图片新特征列...")
    col_name = list(df_image.columns)
    #插入新列之前列名去重
    col_name = col_name + sorted(set(new_features_list) - set(col_name), key = new_features_list.index)
    df_image = df_image.reindex(columns=col_name, fill_value=0)
    logging.info("图片新特征列扩展完成")
    return df_image

def image_feature_extraction(df_image):
    logging.info("开始图片特征提取...")
    #将第三列到最后列转为float
    # df_image.iloc[:,2:] = df_image.iloc[:,2:].astype(float)
    # df_image.iloc[:, -2:] = df_image.iloc[:, -2:].astype(object)
    # return df_image
    df_image['sim_image_word'] = df_image['sim_image_word'].astype(float)
    #其余数据统计
    i = 0
    image_name = []
    for index, row in df_image.iterrows():
        logging.info("处理进度"+str(i+1)+"/"+str(df_image.shape[0]))
        #获得需要处理的文本内容
        if (pd.isna(df_image.iloc[i,1])):
            i += 1
            continue
        else:
            image_list = row['piclist'].split('\t')
            # 计算 颜色矩
            filename1 = 'G:/train/rumor_pic/' + image_list[0]
            filename2 = 'G:/train/truth_pic/' + image_list[0]
            filename= ''
            if (os.path.isfile(filename1)):
                filename = filename1
            else:
                filename = filename2
            #计算颜色矩
            df_image.at[i, 2:11] = image_color_moments(filename)
            #计算深度学习特征 ---PyTorch ResNet50 CNN
            try:
                df_image.at[i, 11:-5] = image_resnet_cnn(filename,model_resnet50)
            except Exception as e:
                logging.info("图片有问题"+str(e))
            df_image['tf_vgg19_class'] = image_get_class(filename)
            # 获得图片的宽度、高度、k物理大小kb
            df_image.at[i, 'image_width'], df_image.at[i, 'image_height'], df_image.at[i, 'image_kb'] = image_get_width_height_kb(filename)
            #计算图文相似度，当存在多张图片的时候采用第一张图片作为该博文的代表图片
            df_image.at[i, 'sim_image_word'] = image_get_img_word_sim(i, row['tf_vgg19_class'], row['tf_resnet50_class'])
            i += 1
    logging.info("图片特征提取结束...")
    return df_image

def image_get_img_word_sim(index, vgg19_class_name, resnet50_class_name):
    """
    similarity_score = arg max{ log( f_i * c_j * swv(term_i,term_j) ) }
    1 ≤ i ≤ n, 1 ≤ j ≤m
    swv(term_i,term_j)即term_i和term_j词向量的余弦相似度
    f_i即第i个词汇(微博正文)的词频
    c_j即第j个词汇(图片分类名)的可信度
    """
    #微博正文
    text_content = df_text['text'][index] 
    #去除停用词和英文单词并分词为list
    list_clear_weibo_text = jieba_clear_text("".join(re.findall(u"[\u4e00-\u9fa5]", text_content))).split(' ')
    #获得微博正文的词频
    dict_weibo_text = Counter(list_clear_weibo_text)
    #获得分类的词向量
    try:
        #获取单词的词向量
        term_vgg19_class_name = model_word2vec[dict_image_class[vgg19_class_name]]
    except Exception:
        #word2vec中不存在这个词汇，以64位0补充
        term_vgg19_class_name = np.zeros(64)
    try:
        #获取单词的词向量
        term_resnet50_class_name = model_word2vec[dict_image_class[resnet50_class_name]]
    except Exception:
        #word2vec中不存在这个词汇，以64位0补充
        term_resnet50_class_name = np.zeros(64)

    list_vgg19_sim = []
    list_resnet50_sim = []
    #遍历微博正文词频表
    for(word, frequency) in dict_weibo_text.items():
        try:
            #获取单词的词向量
            term_i = model_word2vec[word]
        except Exception:
            #word2vec中不存在这个词汇，以64位0补充
            term_i = np.zeros(64)
        if np.all(term_i == 0):
            list_vgg19_sim.append(0)
            list_resnet50_sim.append(0)
            continue
        if np.all(term_vgg19_class_name == 0):
            list_vgg19_sim.append(0)
        if np.all(term_resnet50_class_name == 0):
            list_resnet50_sim.append(0)
        if np.all(term_vgg19_class_name != 0):
            # 计算余弦相似度
            swv_vgg19 = np.dot(term_i, term_vgg19_class_name) / (norm(term_i) * norm(term_vgg19_class_name))
            # 计算图文相似度
            list_vgg19_sim.append(np.log(1 + frequency * float(list_vgg19_score[index]) * swv_vgg19))
        if np.all(term_resnet50_class_name != 0):
            #计算余弦相似度
            swv_resnet50 = np.dot(term_i, term_resnet50_class_name) / (norm(term_i) * norm(term_resnet50_class_name))
            #计算图文相似度
            list_resnet50_sim.append(np.log(1 + frequency*float(list_resnet50_score[index])*swv_resnet50))

    similarity_score = (max(list_vgg19_sim,default=0) + max(list_resnet50_sim,default=0)) / 2
    print(similarity_score)
    return similarity_score

def image_get_score_list(image_class_vgg19_score_path, image_class_resnet50_score_path):
    #获得vgg19和resnet50分类时的可信度
    with open(image_class_vgg19_score_path, "r", encoding='UTF-8') as f1:
        str_vgg19_score = f1.read()
        #分数以空格分开，将str转成list
        list_vgg19_score = str_vgg19_score.split(" ")
    
    with open(image_class_resnet50_score_path, "r", encoding='UTF-8') as f2:
        str_resnet50_score = f2.read()
        #分数以空格分开，将str转成list
        list_resnet50_score = str_resnet50_score.split(" ")
    
    return list_vgg19_score, list_resnet50_score

def image_get_width_height_kb(img_path):
    im = Image.open(img_path)  # 返回一个Image对象
    fsize = os.path.getsize(img_path)
    fsize = fsize / float(1024)
    return im.size[0], im.size[1], round(fsize, 2)

def image_color_moments(filename):
    """
    提取图像颜色矩
    :param filename: 文件路径名
    :return: color_feature：颜色矩特征
    """
    img = cv2.imread(filename)
    if img is None:
        return
    # Convert BGR to HSV colorspace  OpenCV 默认的颜色空间是 BGR，类似于RGB，但不是RGB
    # HSV颜色空间的色调、饱和度、明度与人眼对颜色的主观认识相对比较符合，与其他颜色空间相比HSV空间能更好的反映人类对颜色的感知
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average 一阶矩(均值)
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation 二阶矩(方差)
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness 三阶矩(斜度)
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
    return color_feature

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        # resnet50
        self.net = models.resnet50(pretrained=True)

    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        return output

def image_resnet_cnn(img_path, net):
    transform = transforms.Compose([
        #图片变换为256*256
        transforms.Resize((256,256)),
        #用来将图片从中心裁剪成224*224
        transforms.CenterCrop((224,224)),
        #将图片转成Tensor张量
        transforms.ToTensor()]
    )

    #读入图片并进行统一转换
    img = Image.open(img_path)
    img = transform(img)
    logging.info(img.shape)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    logging.info(x.shape)

    #启用GPU加速
    if torch.cuda.is_available():
        x = x.cuda()
        net = net.cuda()

    #转回CPU，不然可能出错
    y = net(x).cpu()
    y = torch.squeeze(y)
    cnn_features = y.data.numpy().tolist()
    logging.info(y.shape)

    return cnn_features

def image_get_class(img_path):
    img_array = []
    for i in img_path:
        if (i == 'nothing'):
            img_array.append('no')
        else:
            img = image.load_img(i, target_size=(224, 224))
            # 将图片转换为(224,224,3)数组，最后的3是因为RGB三色彩图
            img = image.img_to_array(img)
            # 跟前面的例子一样，使用模型进行预测是批处理模式，
            # 所以对于单个的图片，要扩展一维成为（1,224,224,3)这样的形式
            # 相当于建立一个预测队列，但其中只有一张图片
            img = np.expand_dims(img, axis=0)
            predict_class_vgg = model_tf_vgg19.predict(img)
            # 获取图片识别可能性最高的3个结果
            desc_vgg = vgg19.decode_predictions(predict_class_vgg, top=1)
            # desc = resnet50.decode_predictions(predict_class, top=3)
            # 我们的预测队列中只有一张图片，所以结果也只有第一个有效，显示出来
            img_array.append(desc_vgg[0][0][1])
            print(i)

    # 使用模型预测（识别）
    return img_array

def translateBaidu(text, f='en', t='zh'):
    salt = random.randint(32768, 65536)
    sign = appid + text + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    url = url_baidu + '?appid=' + appid + '&q=' + urllib.parse.quote(text) + '&from=' + f + '&to=' + t + '&salt=' + str(salt) + '&sign=' + sign
    response = urllib.request.urlopen(url)
    content = response.read().decode('utf-8')
    data = json.loads(content)
    result = str(data['trans_result'][0]['dst'])
    return result

def get_cn_json_class(en_imagenet_class_path, cn_imagenet_class_path):
    fn = open(en_imagenet_class_path, "r", encoding='UTF-8')
    j = fn.read()
    dic = json.loads(j) #英文原版
    fn.close()
    txt_dic = {}  #中文
    for i in range(0, 1000):
        try:
            start = time.time()
            txt_dic[dic[str(i)][1]] = translateBaidu(dic[str(i)][1])
            end = time.time()
            if end - start < 1:
                time.sleep(1)  # api接口限制，每秒调用1次
        except Exception as e:
            print(e)
    json_str = json.dumps(txt_dic)
    file_object = open(cn_imagenet_class_path, 'w')
    file_object.write(json_str)
    file_object.close()

def image_get_class_cn_dict(cn_imagenet_class_path):
    """
    获得分类的中文对照词典
    :param cn_imagenet_class_path:
    :return:
    """
    fn = open(cn_imagenet_class_path, "r", encoding='UTF-8')
    str_json = fn.read()
    dic = json.loads(str_json)
    fn.close()
    return dic

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#*******************文本特征提取开始***************************
#原始数据的读入
#df_text,df_user,df_image = train_data_read(train_csv_path)

start = time.time()

# 读入停用词表、积极词汇表、消极词汇表
stopwords = get_stopwords_list()
possentiwords = get_possentiwords_list()
negsentiwords = get_negsentiwords_list()

#文本的读入
df_text = text_data_read()

#微博文本扩展特征数据列
new_text_features_list = ['text_length', 'contains_questmark', 'num_questmarks', 'contains_exclammark',
                     'num_exclammarks', 'contains_hashtag', 'num_hashtags', 'contains_URL',
                     'num_URLs', 'contains_mention', 'num_mentions', 'sentiment_score',
                     'num_noun','num_verb','num_pronoun','num_possentiwords','num_negsentiwords',
                     'contains_firstorderpron','contains_secondorderpron','contains_thirdorderpron']
# 浪费时间
for i in range(1,101):
    new_text_features_list.append('word2vec_'+str(i))
df_text = text_insert_cols(df_text,new_text_features_list)

#加载sentiment model
if not os.path.isfile(sentiment_model_path + '.3'):
    # 情感分析语料模型训练
    text_train_sentiment()
else:
    logging.info("sentiment model is ready!")

#加载word2vec model
if not os.path.isfile(word2vec_model_path):
    # 获得词向量训练语料
    text_get_clear_word2vec_corpus(word2vec_txt_path)
    # 训练word2vec模型
    model_word2vec = text_train_word2vec_model(word2vec_txt_path, word2vec_model_path)
else:
    # 加载word2vec模型
    #model_word2vec = text_load_word2vec_model(word2vec_model_path)
    model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(r'G:\毕设\数据集\微博\news_12g_baidubaike_20g_novel_90g_embedding_64.bin', binary=True)
    remember_delete = 1

#文本特征提取
df_text = text_feature_extraction(df_text)
#文本特征保存
df_text.to_csv(text_csv_path,index=0)#不保留行索引

end = time.time()
logging.info("运行时间："+str(end-start))
#*******************文本特征提取结束***************************


#*******************用户特征提取开始***************************
start = time.time()
#原始数据读入
df_user = user_data_read()
#用户新特征列扩展
new_user_features_list = ['folfans_ratio']
df_user = user_insert_cols(df_user,new_user_features_list)
#用户特征提取
df_user = user_feature_extraction(df_user)
#用户特征保存
df_user.to_csv(user_csv_path,index=0)#不保留行索引

end = time.time()
logging.info("运行时间："+str(end-start))
#*******************用户特征提取结束***************************

#*******************图片特征提取开始***************************

start = time.time()
#原始数据读入
df_image = image_data_read()
#图片新特征列扩展
new_image_features_list = ['h_first_moment','s_first_moment','v_first_moment',
                           'h_second_moment','s_second_moment','v_second_moment',
                           'h_third_moment','s_third_moment','v_third_moment',
                           'tf_vgg19_class','tf_resnet50_class','image_width','image_height','image_kb','sim_image_word']
for i in range(1,2049):
    new_image_features_list.append('resnet_'+str(i))
df_image = image_insert_cols(df_image,new_image_features_list)
#pytorch ResNet 50网络
model_resnet50 = net()
model_resnet50.eval()
model_resnet50 = model_resnet50.cuda()

#tensorflow vgg19和resnet50模型
model_tf_vgg19 = vgg19.VGG19(weights='imagenet')
model_tf_resnet50 = resnet50.ResNet50(weights='imagenet')
model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(r'G:\毕设\数据集\微博\news_12g_baidubaike_20g_novel_90g_embedding_64.bin', binary=True)

#获得vgg19和resnet50分类的图片top1可信度list
list_vgg19_score, list_resnet50_score = image_get_score_list(image_class_vgg19_score_path, image_class_resnet50_score_path)
#获得中文对照词典
dict_image_class = image_get_class_cn_dict(cn_imagenet_class_path)


#获得文本特征中的微博原文
df_text = pd.read_csv(text_csv_path, usecols=['text']) #只加载text列，提升速度，减小不必要的内存损耗

#图片特征提取
df_image = image_feature_extraction(df_image)
#图片特征保存
df_image.to_csv(image_csv_path,index=0)#不保留行索引
end = time.time()
logging.info("运行时间："+str(end-start))
#*******************图片特征提取结束***************************
# 2020-02-09 19:30:23,551 : INFO : 图片有问题Given groups=1, weight of size 64 3 7 7, expected input[1, 1, 224, 224] to have 3 channels, but got 1 channels instead
# Loaded runtime CuDNN library: 7.5.1 but source was compiled with: 7.6.5.  CuDNN library major and minor version needs to match or have higher minor version in case of CuDNN 7.0 or later version. If using a binary install, upgrade your CuDNN library.  If building from sources, make sure the library loaded at runtime is compatible with the version specified during compile configuration.