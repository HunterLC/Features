import pandas as pd
import logging
import time

text_csv_path = r'G:\毕设\数据集\微博\text.csv'
user_csv_path = r'G:\毕设\数据集\微博\user.csv'
image_csv_path = r'G:\毕设\数据集\微博\image.csv'
fusion_csv_path = r'G:\毕设\数据集\微博\fusion_news_features.csv'

def origin_data_read(text_csv_path,user_csv_path,image_csv_path):
    '''
    训练数据的读入
    df_text  文本信息列
    df_user  用户信息列
    df_image 图片信息列
    '''
    logging.info("正在载入文本、用户以及图片数据中...")
    #微博信息
    df_text = pd.read_csv(text_csv_path)
    #用户信息
    df_user = pd.read_csv(user_csv_path)
    #微博图片信息
    df_image = pd.read_csv(image_csv_path)
    logging.info("数据载入完成")
    return df_text,df_user,df_image

def features_fusion(df_text,df_user,df_image):
    logging.info("开始特征融合...")
    df_text = df_text.drop(columns = ['text'], axis = 1)
    df_user = df_user.drop(columns = ['id'], axis = 1)
    df_image = df_image.drop(columns = ['id','piclist'], axis = 1)
    df_result = pd.concat([df_text, df_user,df_image], axis = 1)
    logging.info(df_result.shape)
    return df_result

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
start = time.time()
#原始数据的读入
df_text,df_user,df_image = origin_data_read(text_csv_path,user_csv_path,image_csv_path)
df_result = features_fusion(df_text,df_user,df_image)
df_result.to_csv(fusion_csv_path,index=0)#不保留行索引
end = time.time()
logging.info("运行时间："+str(end-start))