import tensorflow as tf
from tensorflow import keras
# 载入vgg19模型
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse


# 初始化vgg19模型，weights参数指的是使用ImageNet图片集训练的模型
# 每种模型第一次使用的时候都会自网络下载保存的h5文件
# vgg19的数据文件约为584M
model = vgg19.VGG19(weights='imagenet')
# model = resnet50.ResNet50(weights='imagenet')

def image_feature_extraction(df_image):
    #将第三列到最后列转为float
    # df_image.iloc[:,2:] = df_image.iloc[:,2:].astype(float)
    # df_image.iloc[:, -5:-3] = df_image.iloc[:, -5:-3].astype(object)
    # return df_image
    #其余数据统计
    i = 0
    image_name = []
    for index, row in df_image.iterrows():
        #获得需要处理的文本内容
        if (pd.isna(df_image.iloc[i,1])):
            i += 1
            image_name.append('nothing')
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
            # df_image.at[i, 2:11] = image_color_moments(filename)
            #计算深度学习特征 ---PyTorch ResNet50 CNN
            # try:
            #     df_image.at[i, 11:-2] = image_resnet_cnn(filename,model_resnet50)
            # except Exception as e:
            #     logging.info("图片有问题"+str(e))
            # df_image.at[i, 'tf_vgg19_class'] = image_get_class(filename)
            image_name.append(filename)
            i += 1
    df_image['tf_vgg19_class'], img_score = main(image_name)
    return df_image, img_score

def main(imgPath):
    # 载入命令行参数指定的图片文件, 载入时变形为224x224，这是模型规范数据要求的
    img_array = []
    img_score = []
    j = 0
    for i in imgPath:
        if (i == 'nothing'):
            img_array.append('no')
            img_score.append(0)
        else:
            img = image.load_img(i, target_size=(224, 224))
            # 将图片转换为(224,224,3)数组，最后的3是因为RGB三色彩图
            img = image.img_to_array(img)
            # 跟前面的例子一样，使用模型进行预测是批处理模式，
            # 所以对于单个的图片，要扩展一维成为（1,224,224,3)这样的形式
            # 相当于建立一个预测队列，但其中只有一张图片
            img = np.expand_dims(img, axis=0)
            predict_class = model.predict(img)
            # 获取图片识别可能性最高的3个结果
            desc = vgg19.decode_predictions(predict_class, top=1)
            # desc = resnet50.decode_predictions(predict_class, top=3)
            # 我们的预测队列中只有一张图片，所以结果也只有第一个有效，显示出来
            img_array.append(desc[0][0][1])
            img_score.append(desc[0][0][2])
            print(desc[0][0][2])
        j += 1
        print(str(j))
    # x = np.array(img_array)

    # 使用模型预测（识别）
    return img_array, img_score

def image_insert_cols(df_image,new_features_list):
    '''
    增加图片新的特征列，方便后续提取并补充值
    :param df_image: 图片信息
    :return: df_image: 新图片信息dataframe
    '''
    col_name = list(df_image.columns)
    #插入新列之前列名去重
    col_name = col_name + sorted(set(new_features_list) - set(col_name), key = new_features_list.index)
    df_image = df_image.reindex(columns=col_name, fill_value=0)
    return df_image

image_csv_path = r'G:\毕设\数据集\微博\image.csv'
if __name__ == '__main__':
    df_image = pd.read_csv(image_csv_path)
    df_image, img_score = image_feature_extraction(df_image)
    # df_image.to_csv(image_csv_path, index=0)  # 不保留行索引
    str = " ".join('%s' %id for id in img_score)
    file_object = open(r'G:\毕设\数据集\微博\image_class_vgg19.txt', 'w')
    file_object.write(str)
    file_object.close( )