import pandas as pd
import numpy as np

fusion_csv_path = r'G:\毕设\数据集\微博\fusion_news_features.csv'
fusion_no_object_csv_path = r'G:\毕设\数据集\微博\fusion_features_0306_no_object.csv'
new_fusion_csv_path = r'G:\毕设\数据集\微博\fusion_features_0306.csv'
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

def delete_df_object(df):
    gender_map = {'男': 1, '女': 0}

    category_map = {'社会生活': 1, '医药健康': 2, '文体娱乐': 3, '财经商业': 4,
                    '政治': 5, '教育考试': 6, '军事': 7, '科技': 8}

    location_map = {'北京': 1, '广东': 2, '其他': 3, '江苏': 4,
                    '上海': 5, '浙江': 6, '四川': 7, '山东': 8,
                    '河南': 9, '陕西': 10, '海外': 11, '福建': 12,
                    '湖北': 13, '辽宁': 14, '河北': 15, '安徽': 16,
                    '湖南': 17, '重庆': 18, '天津': 19, '江西': 20,
                    '广西': 21, '山西': 22, '黑龙': 23, '吉林': 24,
                    '云南': 25, '贵州': 26, '甘肃': 27, '内蒙': 28,
                    '香港': 29, '海南': 30, '新疆': 31, '台湾': 32,
                    '无': 33, '青海': 34, '宁夏': 35, '西藏': 36,
                    '澳门': 37}
    df['user_gender'] = df['user_gender'].map(gender_map)
    df['category'] = df['category'].map(category_map)
    df['user_location'] = df['user_location'].map(location_map)
    return df

# 特征的预处理
# df = pd.read_csv(fusion_csv_path)
# df = features_preprocessor(df)
# df.to_csv(fusion_csv_path,index=0)#不保留行索引

#特征去除object
df = pd.read_csv(new_fusion_csv_path)
df = delete_df_object(df)
df.to_csv(fusion_no_object_csv_path,index=0)#不保留行索引
