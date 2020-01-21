import numpy as np
import pandas as pd
import re
import os
import time
import jieba
from snownlp import sentiment
from snownlp import SnowNLP
import jieba.posseg as pseg
from gensim.models import word2vec
import logging


train_csv_path = r'G:\毕设\数据集\微博\train.csv'
text_csv_path = r'G:\毕设\数据集\微博\text.csv'
user_csv_path = r'G:\毕设\数据集\微博\user.csv'
train_negative_corpus_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+'/util/negative.txt'
train_positive_corpus_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+'/util/positive.txt'
sentiment_model_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+'/util/sentiment.marshal'
stopwords_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/util/stopwords.txt"
word2vec_txt_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/util/word2vec_corpus.txt"
word2vec_model_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/util/text8.model"

def train_data_read(train_csv_path):
    '''
    训练数据的读入
    df_text  文本信息列
    df_user  用户信息列
    df_image 图片信息列
    '''
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
    col_name = col_name[0:-2]+ new_features_list +col_name[-2:]
    df_text = df_text.reindex(columns=col_name, fill_value=0)
    logging.info("文本新特征列扩展完成")
    return df_text

def text_feature_extraction(df_text):
    logging.info("开始文本特征提取...")
    # #统计字符串长度
    # df_text['text_length'] = df_text['text'].str.len()
    # #将情感分数列转为float
    # df_text['sentiment_score'] = df_text['sentiment_score'].astype(float)
    for j in range(1,101):
        df_text['word2vec_'+str(j)] = df_text['word2vec_'+str(j)].astype(float)
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
        #计算每条微博正文的词向量均值
        df_text.at[i,-102:-2] = text_compute_word2vec(text_content).tolist()
        i += 1
    logging.info("文本特征提取结束...")
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
        if myword not in stopwords:
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

def text_train_word2vec_model(word2vec_txt_path,word2vec_model_path):
    '''
    训练word2vec词向量模型
    :param word2vec_txt_path: 语料路径
    :param word2vec_model_path: 模型保存路径
    :return: 词向量模型
    '''
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
    '''
    加载训练完成的word2vec词向量模型
    :param word2vec_model_path: 模型路径
    :return: 词向量模型
    '''
    model = word2vec.Word2Vec.load(word2vec_model_path)
    return model

def text_get_clear_word2vec_corpus(word2vec_txt_path):
    '''
    从原始微博文本获得word2vec语料文本
    :param word2vec_txt_path: 语料保存位置
    :return: 0
    '''
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
            text_word2vec_score_list.append(model_word2vec.wv[word])
        except KeyError:
            text_word2vec_score_list.append(np.zeros(100))
    result_mean_array = np.mean(np.array(text_word2vec_score_list),axis=0)
    return result_mean_array


def user_data_read():
    '''
    用户特征文件的读取
    :return: 用户特征文件
    '''
    df_user = pd.read_csv(user_csv_path)
    return df_user

def user_insert_cols(df_user,new_features_list):
    '''
    增加用户新的特征列，方便后续提取并补充值
    :param df_user: 用户信息
    :return: df_user: 新用户信息dataframe
    '''
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
    '''
    计算关注/粉丝比
    :param user_follow_count: 关注数
    :param user_fans_count: 粉丝数
    :return:
    '''
    if( user_fans_count == 0):
        return 0
    else:
        return user_follow_count/user_fans_count
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#*******************文本特征提取开始***************************
#*******************版本1*************************************
# start = time.time()
# # 读入停用词表
# stopwords = get_stopwords_list()
# #原始数据的读入
# # df_text,df_user,df_image = train_data_read(train_csv_path)
#
# #微博文本扩展特征数据列
# # new_text_features_list = ['text_length', 'contains_questmark', 'num_questmarks', 'contains_exclammark',
# #                      'num_exclammarks', 'contains_hashtag', 'num_hashtags', 'contains_URL',
# #                      'num_URLs', 'contains_mention', 'num_mentions', 'sentiment_score','num_noun','num_verb','num_pronoun']
# # for i in range(1,101):
# #     new_text_features_list.append('word2vec_'+str(i))
# df_text = text_insert_cols(df_text,new_text_features_list)
# #情感分析语料模型训练
# # text_train_sentiment()
#获得词向量训练语料
# text_get_clear_word2vec_corpus(word2vec_txt_path)
# #训练word2vec模型
# model_word2vec = text_train_word2vec_model(word2vec_txt_path,word2vec_model_path)
#加载word2vec模型
# model_word2vec = text_load_word2vec_model(word2vec_model_path)
#
# df_text = text_feature_extraction(df_text)
# df_text.to_csv(text_csv_path,index=0)#不保留行索引
# end = time.time()
# print("运行时间：",end-start)
# # df_user.to_csv(r'G:\毕设\数据集\微博\user.csv',index=0)#不保留行索引
# # df_image.to_csv(r'G:\毕设\数据集\微博\image.csv',index=0)#不保留行索引

#*******************版本2*************************************
# start = time.time()
#
# # 读入停用词表
# stopwords = get_stopwords_list()
# #原始数据的读入
# df_text = text_data_read()
#
# #微博文本扩展特征数据列
# # new_text_features_list = []
# # for i in range(1,101):
# #     new_text_features_list.append('word2vec_'+str(i))
# # df_text = text_insert_cols(df_text,new_text_features_list)
#
# #情感分析语料模型训练
# # text_train_sentiment()
# #获得词向量训练语料
# # text_get_clear_word2vec_corpus(word2vec_txt_path)
# # #训练word2vec模型
# # model_word2vec = text_train_word2vec_model(word2vec_txt_path,word2vec_model_path)
# #加载word2vec模型
# model_word2vec = text_load_word2vec_model(word2vec_model_path)
#
# df_text = text_feature_extraction(df_text)
# df_text.to_csv(text_csv_path,index=0)#不保留行索引
# end = time.time()
# logging.info("运行时间："+str(end-start))
# # df_user.to_csv(r'G:\毕设\数据集\微博\user.csv',index=0)#不保留行索引
# # df_image.to_csv(r'G:\毕设\数据集\微博\image.csv',index=0)#不保留行索引
#*******************文本特征提取结束***************************


#*******************用户特征提取开始***************************
start = time.time()
#原始数据读入
df_user = user_data_read()
#用户新特征列扩展
# new_user_features_list = ['folfans_ratio']
# df_user = user_insert_cols(df_user,new_user_features_list)
#用户特征提取
df_user = user_feature_extraction(df_user)
#用户特征保存
df_user.to_csv(user_csv_path,index=0)#不保留行索引

end = time.time()
logging.info("运行时间："+str(end-start))
#*******************用户特征提取开始***************************
