# Features
Fake news detection based on feature selection
## 项目树
```
Features
│  .gitignore
│  LICENSE
│  README.md
│  
├─data_statistics
│      statistics.py
│      __init__.py
│      
├─example
│  │  cloud.py
│  │  test.py
│  │  __init__.py
│  │  
│  └─__pycache__
├─image
│  ├─example
│  │      cn_pic.png
│  │      en_pic.png
│  │      mapmask.jpg
│  │      
│  └─feature
│          text_features_1.png
│          text_features_2.png
│          user_features_1.png
│          image_features_color_moments.png
│          
├─test
│  └─example
│          Chinese.txt
│          English.txt
│          
├─util
│      negative.txt
│      positive.txt
│      sentiment.marshal.3
│      stopwords.txt
│      text8.model
│      text_features.xlsx
│      word2vec_corpus.txt
│      
└─work
        feature_extraction.py
        __init__.py
```
## 文本特征提取
 特征名称              | 意义                 | 数据类型   | 备注                                                                                    |
| ------------------- |:--------------------:| --------:|:---------------------------------------------------------------------------------------- |
| id                  | 微博账号标识          |   object  | 无                                                                                       |
| text                | 微博正文              |   object  | 无                                                                                       |
| text_length         | 微博正文长度          |    int    | 无                                                                                      |
| contains_questmark  | 微博正文是否包含 ？    |    int    | 1：包含，0：不包含                                                                        |
| num_questmarks      | ？ 的个数             |    int   | 无                                                                                       |
| contains_exclammark | 微博正文是否包含 ！    |    int   | 1：包含，0：不包含                                                                        |
| num_exclammarks     | ！ 的个数             |    int   | 无                                                                                      |
| contains_hashtag    | 微博正文是否包含话题   |    int   | 1：包含，0：不包含                                                                        |
| num_hashtags        | 话题个数              |    int   | 新浪微博话题格式为两个#，例如 #春节#                                                       |
| contains_URL        | 微博正文是否包含链接   |    int   | 1：包含，0：不包含                                                                        |
| num_URLs            | 链接个数              |    int   | 链接需包含http头的，例如http://www.baidu.com 计数1；不包含http头的例如www.baidu.com不计数     |
| contains_mention    | 微博正文是否包含提及@  |    int   | 1：包含，0：不包含                                                                        |
| num_mentions        | @ 的个数             |    int    | 无                                                                                      |
| sentiment_score     | 情感分数             |    float  | 使用snownlp打分，取值0~1之间，越接近0越消极，越接近1越积极                                   |
| num_noun            | 名词个数             |    int    | 无                                                                                      |
| num_verb            | 动词个数             |    int    | 无                                                                                      |
| num_pronoun         | 代词个数             |    int    | 无                                                                                      |
| word2vec_1~100      | 词向量列1~100        |    float  | 采用word2vec构建词向量，每个正文利用jieba分词后计算词矩阵之和，并取平均值作为文本的词向量       |
| category            | 微博新闻类属         |    object | 无                                                                                      |
| label               | 真假新闻标签         |    int    | 0：真新闻 ，1:假新闻                                                                      |
| num_possentiwords   | 积极词汇个数         |    int    | 无                                                                                      |
| num_negsentiwords   | 消极词汇个数         |    int    | 无                                                                                      |

![文本特征截图1 ''文本内容''](https://github.com/HunterLC/Features/blob/master/image/feature/text_features_1.png)
![文本特征截图2 ''词向量''](https://github.com/HunterLC/Features/blob/master/image/feature/text_features_2.png)

## 用户特征提取
 特征名称            | 意义            |   类型      | 备注       |
| -------------     |:---------------:| ----------:|:------     |
| id                | 微博账号标识     |   object   | 无          |
| user_gender       | 用户性别        |    object   | 男、女     |
| user_follow_count | 用户关注数      |    float    | 无         |
| user_fans_count   | 用户粉丝数      |    float    | 无         |
| user_weibo_count  | 用户微博数      |    float    | 无         |
| folfans_ratio     | 关注/粉丝比     |    float    | 无         |
| user_location     | 用户所在地      |    float    | 无         |
| user_description  | 用户个性签名    |    float    | 无         |

![用户特征截图1](https://github.com/HunterLC/Features/blob/master/image/feature/user_features_1.png)

## 图片特征提取
### 颜色特征
颜色矩 Color Moments
> 在颜色特征方面，为减小运算量，采用颜色矩作为检索的特征，提取被检测图像的一、二、三阶矩。由于HSV颜色空间的色调、饱和度、明度与人眼对颜色的主观认识相对比较符合，与其他颜色空间相比HSV空间能更好的反映人类对颜色的感知。颜色信息集中在图像颜色的低阶矩中，主要是每种颜色分量的一阶矩(均值)、二阶矩(方差)、三阶矩(斜度)，可以有效地表示图像中的颜色分布。
> 
>hue（色调）、saturation（饱和度）、value（明度）
>
 特征名称            | 意义           |   类型      | 备注       |
| -------------     |:--------------:| ----------:|:------     |
| h_first_moment    | 色相一阶矩      |   float   | 无          |
| s_first_moment    | 饱和度一阶矩    |    float   | 无          |
| v_first_moment    | 亮度一阶矩      |    float    | 无         |
| h_second_moment   | 色相二阶矩      |    float    | 无         |
| s_second_moment   | 饱和度二阶矩    |    float    | 无         |
| v_second_moment   | 亮度二阶矩      |    float    | 无         |
| h_third_moment    | 色相三阶矩      |    float    | 无         |
| s_third_moment    | 饱和度三阶矩    |    float    | 无         |
| v_third_moment    | 亮度三阶矩      |    float    | 无         |

![图片特征截图1](https://github.com/HunterLC/Features/blob/master/image/feature/image_features_color_moments.png)

### 深度学习特征
基于PyTorch的ResNet50
> 这块不太会
>
 特征名称            | 意义           |   类型      | 备注       |
| -------------     |:--------------:| ----------:|:------     |
| resnet_1~2048    | resnet50特征      |   float   | 无          |

![图片特征截图1](https://github.com/HunterLC/Features/blob/master/image/feature/image_features_resnet50.png)

## 预测结果
由于内存小，跑完数据集会爆，所以选了真假新闻各4000个
跑了下K-近邻分类算法，没有对数据集做任何预处理和特征选择的结果如下
![预测截图1](https://github.com/HunterLC/Features/blob/master/image/feature/result.png)

没有对数据集做任何预处理和特征选择的决策树效果如下
> model_dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=4,min_samples_split=6)
>
![预测截图2](https://github.com/HunterLC/Features/blob/master/image/feature/result_dt.png)

没有对数据集做任何预处理和特征选择的随机森林效果如下
> RandomForestClassifier(n_estimators=100,
                                   bootstrap=True,
                                   max_features='sqrt')
>
![预测截图3](https://github.com/HunterLC/Features/blob/master/image/feature/result_rf.png)

## 更新日志
### 2020-03-01
1.对数据集进行初步的预处理

### 2020-02-27
1.解决图文相似度特征部分结果为nan的bug,原因在于计算向量的余弦相似度swv时存在负值，导致log计算出现非法参数

![图文相似度1](https://github.com/HunterLC/Features/blob/master/image/feature/sim_image_word_1.png)

2.测试( 文本 + 用户 + 图片 )特征预测结果

![预测结果0227](https://github.com/HunterLC/Features/blob/master/image/feature/rf_0227_all.png)

>随机森林准确率：
 0.941306405806248
>
3.测试( 文本 )特征预测结果

![预测结果0227](https://github.com/HunterLC/Features/blob/master/image/feature/rf_0227_text.png)

>随机森林准确率：
 0.9510886715052067
>
4.测试( 用户 )特征预测结果

![预测结果0227](https://github.com/HunterLC/Features/blob/master/image/feature/rf_0227_user.png)

>随机森林准确率：
 0.8982854738613653
>
4.测试( 文本 + 用户 )特征预测结果

![预测结果0227](https://github.com/HunterLC/Features/blob/master/image/feature/rf_0227_text_user.png)

5.测试( 图片 )特征预测结果
包含两个分类'tf_vgg19_class','tf_resnet50_class'时

![预测结果0227](https://github.com/HunterLC/Features/blob/master/image/feature/rf_0227_image_1.png)

不包含两个分类'tf_vgg19_class','tf_resnet50_class'时

![预测结果0227](https://github.com/HunterLC/Features/blob/master/image/feature/rf_0227_image_2.png)

6.测试( 用户 + 图片 )特征预测结果 

![预测结果0227](https://github.com/HunterLC/Features/blob/master/image/feature/rf_0227_user_image.png)

### 2020-02-26
1.新增图文相似度特征，用以表示图像和文本的相似度
>similarity_score = arg max{ log( f_i * c_j * swv(term_i,term_j) ) }
>
>    1 ≤ i ≤ n, 1 ≤ j ≤m
>
>    swv(term_i,term_j)即term_i和term_j词向量的余弦相似度
>
>    f_i即第i个词汇(微博正文)的词频
>
>    c_j即第j个词汇(图片分类名)的可信度
>
 特征名称                  | 意义             |   类型    | 备注                 |
| :--------------------:  |:---------------:| :--------:|:-------------------:|
| sim_image_word          | 图文相似度  |   float        |采用词嵌入方式，减少中英翻译误差中的影响|

![图文相似度](https://github.com/HunterLC/Features/blob/master/image/feature/sim_image_word.png)

目前看还存在一些问题，利用log(x)计算，部分值被存储为nan


### 2020-02-25
1.新增文本特征，统计新闻是否出现第一人称、第二人称、第三人称

 特征名称                  | 意义             |   类型    | 备注                 |
| :--------------------:  |:---------------:| :--------:|:-------------------:|
| contains_firstorderpron | 是否包含第一人称  |   int     | 1:有，0：无          |
| contains_secondorderpron| 是否包含第二人称  |   int     | 1:有，0：无          |
| contains_thirdorderpron | 是否包含第三人称  |   int     | 1:有，0：无          |

2.测试新增加的特征，同时删除了400+行数据缺失严重的行，结果显示10轮平均F1在0.91，最后稳定在0.86附近
![预测截图4](https://github.com/HunterLC/Features/blob/master/image/feature/result_20200225.png)


### 2020-02-23
1.添加图片的宽度、高度、物理大小

2.添加tensorflow的vgg19、resnet50进行图片分类

 特征名称         | 意义          |   类型     | 备注       |
| :------------:  |:------------:| :------------:|:------:|
| tf_vgg19_class | vgg19分类     |   object     | 无          |
| tf_resnet_class| resnet50分类  |   object     | 无          |
| image_width    | 图片宽度      |   int     | 无          |
| image_height   | 图片高度      |   int     | 无          |
| image_kb       | 图片物理大小  |   float   | 单位为kb     |

### 2020-02-19
由于snownlp的情感分数效果不太理想，故采取方案a来补充情感分数部分(b方案需要标记数据集语料，同时过于复杂，暂时舍弃)
>a.增加诸如积极词汇数量，消极词汇数量这类统计特征，对文本的情感信息进行提取，补充文本特征集。
b.尝试其他nlp模型，例如结合循环神经网络的情感分析模型可能具有较好的效果。
>
