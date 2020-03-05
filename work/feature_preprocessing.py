import pandas as pd
import numpy as np

fusion_csv_path = r'G:\毕设\数据集\微博\fusion_news_features.csv'
text_csv_path = r'G:\毕设\数据集\微博\text.csv'
user_csv_path = r'G:\毕设\数据集\微博\user.csv'
image_csv_path = r'G:\毕设\数据集\微博\image.csv'

def get_save_index():
    df_user = pd.read_csv(user_csv_path, usecols=['user_gender'])
    # 保留user_gender列中的非空行，非空为True，空行为False
    save_index = df_user.isnull().sum(axis=1) == 0
    return save_index

def features_preprocessor(df):
    #获取需要保留的index行
    save_index = get_save_index()
    #剔除 455 行用户特征缺少严重的行
    df = df[save_index]
    #用户数据预处理
    df['user_follow_count'].fillna(df['user_follow_count'].mean(), inplace=True)#平均值补充用户关注数
    df['user_fans_count'].fillna(df['user_fans_count'].mean(), inplace=True)  # 平均值补充用户粉丝数
    df['user_weibo_count'].fillna(0, inplace=True)  # 0补充用户发博数
    df['user_location'].fillna('无', inplace=True)#用户地址缺失用 无 替代
    df['user_location'] = df['user_location'].apply(lambda x:x[0:2])# 用户地址统一用前两个字表示
    i = 0
    for index, row in df.iterrows():
        #关注粉丝比
        try:
            df.at[i, 'folfans_ratio'] = row['user_follow_count'] / row['user_fans_count']
        except:
            df.at[i, 'folfans_ratio'] = 0
            #用户描述，0代表没有描述，1代表有描述
        if pd.isna(row['user_description']):
            df.at[i, 'user_description'] = 0
        else:
            df.at[i, 'user_description'] = 1
        i += 1
    # #图片数据预处理(暂无)
    return df

df = pd.read_csv(fusion_csv_path)
df = features_preprocessor(df)
df.to_csv(fusion_csv_path,index=0)#不保留行索引