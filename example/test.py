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

# import cv2
# import numpy as np
#
# # Compute low order moments(1,2,3)
# def color_moments(filename):
#     img = cv2.imread(filename)
#     if img is None:
#         return
#     # Convert BGR to HSV colorspace  OpenCV 默认的颜色空间是 BGR，类似于RGB，但不是RGB
#     # HSV颜色空间的色调、饱和度、明度与人眼对颜色的主观认识相对比较符合，与其他颜色空间相比HSV空间能更好的反映人类对颜色的感知
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     # Split the channels - h,s,v
#     h, s, v = cv2.split(hsv)
#     # Initialize the color feature
#     color_feature = []
#     # N = h.shape[0] * h.shape[1]
#     # The first central moment - average 一阶矩(均值)
#     h_mean = np.mean(h)  # np.sum(h)/float(N)
#     s_mean = np.mean(s)  # np.sum(s)/float(N)
#     v_mean = np.mean(v)  # np.sum(v)/float(N)
#     color_feature.extend([h_mean, s_mean, v_mean])
#     # The second central moment - standard deviation 二阶矩(方差)
#     h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
#     s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
#     v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
#     color_feature.extend([h_std, s_std, v_std])
#     # The third central moment - the third root of the skewness 三阶矩(斜度)
#     h_skewness = np.mean(abs(h - h.mean())**3)
#     s_skewness = np.mean(abs(s - s.mean())**3)
#     v_skewness = np.mean(abs(v - v.mean())**3)
#     h_thirdMoment = h_skewness**(1./3)
#     s_thirdMoment = s_skewness**(1./3)
#     v_thirdMoment = v_skewness**(1./3)
#     color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
#     return color_feature
# print(color_moments(r'G:\毕设\数据集\微博\train\truth_pic\000ffb2483e3e1bd634e920fa2cc93b9.jpg'))
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

# import random
#
# import numpy as np
# import tensorflow as tf
# from sklearn import svm
# from sklearn import preprocessing
#
# right0 = 0.0  # 记录预测为1且实际为1的结果数
# error0 = 0  # 记录预测为1但实际为0的结果数
# right1 = 0.0  # 记录预测为0且实际为0的结果数
# error1 = 0  # 记录预测为0但实际为1的结果数
#
# for file_num in range(10):
#     # 在十个随机生成的不相干数据集上进行测试，将结果综合
#     print('testing NO.%d dataset.......' % file_num)
#     ff = open('digit_train_' + file_num.__str__() + '.data')
#     rr = ff.readlines()
#     x_test2 = []
#     y_test2 = []
#
#     for i in range(len(rr)):
#         x_test2.append(list(map(int, map(float, rr[i].split(' ')[:256]))))
#         y_test2.append(list(map(int, rr[i].split(' ')[256:266])))
#     ff.close()
#     # 以上是读出训练数据
#     ff2 = open('digit_test_' + file_num.__str__() + '.data')
#     rr2 = ff2.readlines()
#     x_test3 = []
#     y_test3 = []
#     for i in range(len(rr2)):
#         x_test3.append(list(map(int, map(float, rr2[i].split(' ')[:256]))))
#         y_test3.append(list(map(int, rr2[i].split(' ')[256:266])))
#     ff2.close()
#     # 以上是读出测试数据
#
#     sess = tf.InteractiveSession()
#
#
#     # 建立一个tensorflow的会话
#
#     # 初始化权值向量
#     def weight_variable(shape):
#         initial = tf.truncated_normal(shape, stddev=0.1)
#         return tf.Variable(initial)
#
#
#     # 初始化偏置向量
#     def bias_variable(shape):
#         initial = tf.constant(0.1, shape=shape)
#         return tf.Variable(initial)
#
#
#     # 二维卷积运算，步长为1，输出大小不变
#     def conv2d(x, W):
#         return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
#     # 池化运算，将卷积特征缩小为1/2
#     def max_pool_2x2(x):
#         return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#
#     # 给x，y留出占位符，以便未来填充数据
#     x = tf.placeholder("float", [None, 256])
#     y_ = tf.placeholder("float", [None, 10])
#     # 设置输入层的W和b
#     W = tf.Variable(tf.zeros([256, 10]))
#     b = tf.Variable(tf.zeros([10]))
#     # 计算输出，采用的函数是softmax（输入的时候是one hot编码）
#     y = tf.nn.softmax(tf.matmul(x, W) + b)
#     # 第一个卷积层，5x5的卷积核，输出向量是32维
#     w_conv1 = weight_variable([5, 5, 1, 32])
#     b_conv1 = bias_variable([32])
#     x_image = tf.reshape(x, [-1, 16, 16, 1])
#     # 图片大小是16*16，,-1代表其他维数自适应
#     h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
#     h_pool1 = max_pool_2x2(h_conv1)
#     # 采用的最大池化，因为都是1和0，平均池化没有什么意义
#
#     # 第二层卷积层，输入向量是32维，输出64维，还是5x5的卷积核
#     w_conv2 = weight_variable([5, 5, 32, 64])
#     b_conv2 = bias_variable([64])
#
#     h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
#     h_pool2 = max_pool_2x2(h_conv2)
#
#     # 全连接层的w和b
#     w_fc1 = weight_variable([4 * 4 * 64, 256])
#     b_fc1 = bias_variable([256])
#     # 此时输出的维数是256维
#     h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 64])
#     h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
#     # h_fc1是提取出的256维特征，很关键。后面就是用这个输入到SVM中
#
#     # 设置dropout，否则很容易过拟合
#     keep_prob = tf.placeholder("float")
#     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
#     # 输出层，在本实验中只利用它的输出反向训练CNN，至于其具体数值我不关心
#     w_fc2 = weight_variable([256, 10])
#     b_fc2 = bias_variable([10])
#
#     y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
#     cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
#     # 设置误差代价以交叉熵的形式
#     train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#     # 用adma的优化算法优化目标函数
#     correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#     sess.run(tf.global_variables_initializer())
#     for i in range(3000):
#         # 跑3000轮迭代，每次随机从训练样本中抽出50个进行训练
#         batch = ([], [])
#         p = random.sample(range(795), 50)
#         for k in p:
#             batch[0].append(x_test2[k])
#             batch[1].append(y_test2[k])
#         if i % 100 == 0:
#             train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
#             # print "step %d, train accuracy %g" % (i, train_accuracy)
#         train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.6})
#         # 设置dropout的参数为0.6，测试得到，大点收敛的慢，小点立刻出现过拟合
#
#     print("test accuracy %g" % accuracy.eval(feed_dict={x: x_test3, y_: y_test3, keep_prob: 1.0}))
#     # def my_test(input_x):
#     #     y = tf.nn.softmax(tf.matmul(sess.run(x), W) + b)
#
#     for h in range(len(y_test2)):
#         if np.argmax(y_test2[h]) == 7:
#             y_test2[h] = 1
#         else:
#             y_test2[h] = 0
#     for h in range(len(y_test3)):
#         if np.argmax(y_test3[h]) == 7:
#             y_test3[h] = 1
#         else:
#             y_test3[h] = 0
#     # 以上两步都是为了将源数据的one hot编码改为1和0，我的学号尾数为7
#     x_temp = []
#     for g in x_test2:
#         x_temp.append(sess.run(h_fc1, feed_dict={x: np.array(g).reshape((1, 256))})[0])
#     # 将原来的x带入训练好的CNN中计算出来全连接层的特征向量，将结果作为SVM中的特征向量
#     x_temp2 = []
#     for g in x_test3:
#         x_temp2.append(sess.run(h_fc1, feed_dict={x: np.array(g).reshape((1, 256))})[0])
#
#     clf = svm.SVC(C=0.9, kernel='linear')  # linear kernel
#     #    clf = svm.SVC(C=0.9, kernel='rbf')   #RBF kernel
#     # SVM选择了RBF核，C选择了0.9
#     #    x_temp = preprocessing.scale(x_temp)  #normalization
#
#     clf.fit(x_temp, y_test2)
#     # SVM选择了RBF核，C选择了0.9
#     print('svm testing accuracy:')
#     print(clf.score(x_temp2, y_test3))
#
#     for j in range(len(x_temp2)):
#         # 验证时出现四种情况分别对应四个变量存储
#         # 这里报错了,需要对其进行reshape（1，-1）
#         if clf.predict(x_temp2[j].reshape(1, -1))[0] == y_test3[j] == 1:
#             right0 += 1
#         elif clf.predict(x_temp2[j].reshape(1, -1))[0] == y_test3[j] == 0:
#             right1 += 1
#         elif clf.predict(x_temp2[j].reshape(1, -1))[0] == 1 and y_test3[j] == 0:
#             error0 += 1
#         else:
#             error1 += 1
#
# accuracy = right0 / (right0 + error0)  # 准确率
# recall = right0 / (right0 + error1)  # 召回率
# print('svm right ratio ', (right0 + right1) / (right0 + right1 + error0 + error1))
# print('accuracy ', accuracy)
# print('recall ', recall)
# print('F1 score ', 2 * accuracy * recall / (accuracy + recall))  # 计算F1值

# from PIL import Image
# import os
# filename = r'G:\train\rumor_pic\0afa91cdde95373b8e4e88daeae7f815.jpg'
# im = Image.open(filename)#返回一个Image对象
# print('宽：%d,高：%d'%(im.size[0],im.size[1]))
# fsize = os.path.getsize(filename)
# fsize = fsize / float(1024)
# print(round(fsize,2))
# import hashlib
# import time
# import urllib
# import random
#
# from gensim.models import word2vec
# import gensim
# import jieba
# import numpy as np
# from scipy.linalg import norm
# from translate import Translator
# import json
#
# # model_file = r'G:\毕设\数据集\微博\news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
# # model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
# # translator= Translator(from_lang="english",to_lang="chinese")
# # translation = translator.translate(dic[str(i)][1])
# #
# appid = '20190716000318328'
# secretKey = '7pjdBCkaUodI5eNqsBWB'
# url_baidu = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
# def translateBaidu(text, f='en', t='zh'):
#     salt = random.randint(32768, 65536)
#     sign = appid + text + str(salt) + secretKey
#     sign = hashlib.md5(sign.encode()).hexdigest()
#     url = url_baidu + '?appid=' + appid + '&q=' + urllib.parse.quote(text) + '&from=' + f + '&to=' + t + '&salt=' + str(salt) + '&sign=' + sign
#     response = urllib.request.urlopen(url)
#     content = response.read().decode('utf-8')
#     data = json.loads(content)
#     result = str(data['trans_result'][0]['dst'])
#     return result
#
# fn = open(r'G:\毕设\数据集\微博\imagenet_class_cn.json', "r", encoding='UTF-8')
# j = fn.read()
# dic = json.loads(j)
# fn.close()
# print(dic)
# txt_dic = {}
# for i in range(0,1000):
#     try:
#         start = time.time()
#         txt_dic[dic[str(i)][1]] = translateBaidu(dic[str(i)][1])
#         end = time.time()
#         if end - start < 1:
#             time.sleep(1) #api接口限制，每秒调用1次
#     except Exception as e:
#         print(e)
# json_str = json.dumps(txt_dic)
# file_object = open(r'G:\毕设\数据集\微博\imagenet_class_cn.json', 'w')
# file_object.write(json_str)
# file_object.close( )
# # translateBaidu('borzoi')

# list = [0,1.02365666,2.66666]
# str = " ".join('%s' %id for id in list)
# print(str)

# import pandas as pd
# import numpy as np
# text_csv_path = r'C:\Backup\桌面\test.csv'
# df = pd.read_csv(text_csv_path) #只加载text列，提升速度，减小不必要的内存损耗

# aa = ['a','b','label']
# df_list = [x for x in aa if x not in ['label']]
# print(df_list)


# df1 = pd.read_csv(r"G:/1.csv")
# df2 = pd.read_csv(r"G:/2.csv")
# for i in df1.columns:
#     for j in df2.columns:
#         if df1[i].to_list() == df2[j].to_list():
#             print(i)
#             break


with open(r'E:\PythonCode\Features\util\word2vec_corpus.txt', "r", encoding='UTF-8') as f:
    chinese_text = f.readlines()
    text_1 = chinese_text[0:19188]
    print(len(text_1))
    text_2 = chinese_text[19188:]
    print(len(text_2))
    text_2_new =[]
    for item in text_2:
        if item.find('锦绣') == -1:
            text_2_new.append(item)
    print(len(text_2_new))