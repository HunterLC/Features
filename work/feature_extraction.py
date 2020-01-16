import numpy as np
import pandas as pd
import re

train_csv_path = r'G:\毕设\数据集\微博\train.csv'

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

def text_insert_cols(df_text):
    '''
    增加文本新的特征列，方便后续提取并补充值
    :param df_text: 文本信息
    :return: df_text: 新文本信息dataframe
    '''
    print("正在扩展文本新特征列...")
    col_name = list(df_text.columns)
    new_features_name = ['text_length','contains_questmark','num_questmarks','contains_exclammark',
                         'num_exclammarks','contains_hashtag','num_hashtags','contains_URL',
                         'num_URLs','contains_mention','num_mentions','num_possentiwords','num_negsentiwords']
    col_name = col_name[0:2]+ new_features_name +col_name[2:]
    df_text = df_text.reindex(columns=col_name, fill_value=0)
    print("文本新特征列扩展完成...")
    return df_text

def text_feature_extraction(df_text):
    print("开始文本特征提取...")
    #统计字符串长度
    df_text['text_length'] = df_text['text'].str.len()
    #其余数据统计
    i = 0
    for index, row in df_text.iterrows():
        print("处理进度",i+1,"/",df_text.shape[0])
        #获得需要处理的文本内容
        text_content = row['text']
        #获得是否含有问号以及问号的数量
        df_text.at[i,'contains_questmark'], df_text.at[i,'num_questmarks'] = text_questmark(text_content)
        #获得是否含有感叹号以及感叹号的数量
        df_text.at[i, 'contains_exclammark'], df_text.at[i, 'num_exclammarks'] = text_exclammark(text_content)
        #获得是否含有hashtag以及hashtag的数量
        df_text.at[i, 'contains_hashtag'], df_text.at[i, 'num_hashtags'] = text_hashtag(text_content)
        #获得是否含有url以及url的数量
        df_text.at[i, 'contains_URL'], df_text.at[i, 'num_URLs'] = text_url(text_content)
        i += 1
    print("文本特征提取结束...")
    return df_text

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

df_text,df_user,df_image = train_data_read(train_csv_path)
df_text = text_insert_cols(df_text)
df_text = text_feature_extraction(df_text)
df_text.to_csv(r'G:\毕设\数据集\微博\text.csv',index=0)#不保留行索引
# df_user.to_csv(r'G:\毕设\数据集\微博\user.csv',index=0)#不保留行索引
# df_image.to_csv(r'G:\毕设\数据集\微博\image.csv',index=0)#不保留行索引