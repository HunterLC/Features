# s = '??？？?？'
# print(s.count("?"))
# print(s.count("？"))

# a = 5
# b = 2
# print(a/b)

# import re
#
# def Find(string):
#     url = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', string)
#     # url = regex.findall(string)
#     return url
#
# string = 'Runoob 的网页地址为：https://www.runoob.com，Google 的网页地址为：https://www.google.com,哈哈哈 http://t.cn/RqFEaUT 态度网(www.taiduw.com)  拍摄者直呼上帝啊。http://www.miaopai.com/show/~ZNc8qjCvIyVurnl4oQAkQ__.htm'
# print("Urls: ", Find(string))


# from snownlp import sentiment
# from snownlp import SnowNLP
#
# # sentiment.train(r'E:\PythonCode\Features\util\negative.txt',r'E:\PythonCode\Features\util\positive.txt')
# # sentiment.save(r'E:\PythonCode\Features\util\sentiment.marshal')
# s = SnowNLP('那个人真让人不舒服 ')
# print(s.sentiments)

# import jieba.posseg as pseg
# import re
# words =pseg.cut("".join(re.findall(u"[\u4e00-\u9fa5]","【外卖配送员无证驾驶闯红灯924次面临十几万罚款】近日，广东茂名民警在对一辆闯红灯的外卖配送摩托进行核查时大吃一惊：该车仅半年就闯红灯924次！平均每天五六次，最多时一天闯红灯十八次。按道路交通安全法，该驾驶员要为他900多次闯红灯的行为缴纳十几万元的罚款。中国新闻网...（中国新闻网）")))
# n = 0
# r = 0
# v = 0
# for w in words:
#     print(w.word,w.flag)
#     if(w.flag.startswith('n')):
#         n += 1
#     elif(w.flag.startswith('v')):
#         v += 1
#     elif(w.flag.startswith('r')):
#         r += 1
# print("名词个数：",n)
# print("动词个数：",v)
# print("代词个数：",r)
# from gensim.models import word2vec
# import logging
# import jieba
# import os
# import re
# import pandas as pd
# import jieba.posseg as pseg
# text_csv_path = r'G:\毕设\数据集\微博\text.csv'
# text_raw_path= r'G:\毕设\数据集\微博\test.txt'
# stopwords_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/util/stopwords.txt"
#
# def text_data_read():
#     '''
#     文本特征文件的读取
#     :return: 文本特征文件
#     '''
#     df_text = pd.read_csv(text_csv_path)
#     return df_text
#
# def jieba_clear_text(text):
#     '''
#     jieba分词，并使用自定义停用词表去除停用词
#     '''
#     raw_result = "/".join(jieba.cut(text))
#     myword_list = []
#     #去除停用词
#     for myword in raw_result.split('/'):
#         if myword not in stopwords:
#             myword_list.append(myword)
#     return " ".join(myword_list)
#
# def get_stopwords_list():
#     '''
#     获得停用词的列表
#     :return: stopwords：停用词列表
#     '''
#     my_stopwords = []
#     fstop = open(stopwords_path, "r", encoding='UTF-8')
#     for eachWord in fstop.readlines():
#         my_stopwords.append(eachWord.strip())
#     fstop.close()
#     return my_stopwords
#
# def text_train_word2vec_model(raw_txt):
#     logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#     sentences = word2vec.Text8Corpus(raw_txt)
#     model = word2vec.Word2Vec(sentences,size=100, window=5, min_count=5, negative=3, sample=0.001, hs=1,workers=4)
#     model.save(r'E:\PythonCode\Features\util\text8.model')
#     return model
#
# def text_load_word2vec_model():
#     model = word2vec.Word2Vec.load(r'G:\毕设\数据集\微博\news_12g_baidubaike_20g_novel_90g_embedding_64.model')
#     return model
#
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# # stopwords = get_stopwords_list()
# # df_text = text_data_read()
# # txt = ''
# # with open(r'G:\毕设\数据集\微博\test.txt', 'a') as f:
# #     for index, row in df_text.iterrows():
# #         text_content = row['text']
# #         raw_txt = jieba_clear_text("".join(re.findall(u"[\u4e00-\u9fa5]", text_content)))
# #         f.write(raw_txt+"\n")
# # logging.info("运行结束")
#
#
# # sentences = word2vec.Text8Corpus(text_raw_path)
# # model = word2vec.Word2Vec(sentences,size=100,workers=4)
# # model.save(r'E:\PythonCode\Features\util\text8.model')
# model = text_load_word2vec_model()

# import numpy as np
# a = np.array([1,1,1,1])
# b = np.array([2,2,2,2])
# c = a+b
# print(c)
# list = []
# list.append(a)
# list.append(b)
# c = np.array(list)
# d = np.mean(c,axis=0)
# print(d.tolist())
# print(d)

# s = 'hahha 555 jck 98 -- *'.split(' ')
# for word in s:
#     print(word)

import cv2
import numpy as np

# Compute low order moments(1,2,3)
def color_moments(filename):
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
print(color_moments(r'G:\毕设\数据集\微博\train\truth_pic\000ffb2483e3e1bd634e920fa2cc93b9.jpg'))
# print(color_moments('E:\PythonCode\Features\image\example\cn_pic.png'))

# list_a = [1,2,3,4,5]
# list_b = [4,5,6,7,8]
# list_c = sorted(set(list_b) - set(list_a),key=list_b.index)
# import numpy as np
# import pandas as pd
# import logging
# image_csv_path = r'G:\毕设\数据集\微博\image.csv'
# def image_data_read():
#     '''
#     图片特征文件的读取
#     :return: 图片特征文件
#     '''
#     df_image = pd.read_csv(image_csv_path)
#     return df_image
#
# def image_insert_cols(df_image,new_features_list):
#     '''
#     增加图片新的特征列，方便后续提取并补充值
#     :param df_image: 图片信息
#     :return: df_image: 新图片信息dataframe
#     '''
#     logging.info("正在扩展图片新特征列...")
#     col_name = list(df_image.columns)
#     #插入新列之前列名去重
#     col_name = col_name + sorted(set(new_features_list) - set(col_name), key = new_features_list.index)
#     df_image = df_image.reindex(columns=col_name, fill_value=0)
#     logging.info("图片新特征列扩展完成")
#     return df_image
# #原始数据读入
# df_image = image_data_read()
# #图片新特征列扩展
# new_image_features_list = ['h_first_moment','s_first_moment','v_first_moment',
#                            'h_second_moment','s_second_moment','v_second_moment',
#                            'h_third_moment','s_third_moment','v_third_moment']
# df_image = image_insert_cols(df_image,new_image_features_list)