import numpy as np
import pandas as pd

train_csv_path = r'G:\毕设\数据集\微博\train.csv'

def train_data_read(train_csv_path):
    '''
    训练数据的读入
    df_text  文本信息列
    df_user  用户信息列
    df_image 图片信息列
    '''
    #微博信息
    df_text = pd.read_csv(train_csv_path,usecols=['id','text','category','label'])   
    #用户信息
    df_user = pd.read_csv(train_csv_path,usecols=['id','userGender','userFollowCount','userFansCount','userWeiboCount','userLocation','userDescription'])
    #微博图片信息
    df_image = pd.read_csv(train_csv_path,usecols=['id','piclist'])
    return df_text,df_user,df_image

def text_insert_cols(df_text):
    '''
    增加文本新的特征列，方便后续提取并补充值
    :param df_text: 文本信息
    :return: df_text: 新文本信息dataframe
    '''
    return df_text
df_text,df_user,df_image = train_data_read(train_csv_path)
df_text.to_csv(r'G:\毕设\数据集\微博\text.csv',index=0)#不保留行索引
df_user.to_csv(r'G:\毕设\数据集\微博\user.csv',index=0)#不保留行索引
df_image.to_csv(r'G:\毕设\数据集\微博\image.csv',index=0)#不保留行索引