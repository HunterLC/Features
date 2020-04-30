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
### 2020-04-30
1.测试特征选择算法选择出的特征子集所训练模型分类的时间和性能
```
随机森林ACC：0.9430018999366688
随机森林F 1：0.9429841556647153
随机森林AUC：0.943108385285605
耗时：8.412
```

2.测试原始数据集
```
随机森林ACC：0.9039476461895715
随机森林F 1：0.9037340673465216
随机森林AUC：0.9042079111728462
耗时：158.66994190216064
```


### 2020-04-27
1.添加性别与真假新闻发布样本数之间的关系图
2.绘制CDF曲线
### 2020-04-19
1.添加特征相关性热力图

![特征相关性热力图](https://github.com/HunterLC/Features/blob/master/image/feature/feature_corr.png)

2.重新绘制特征重要性分数图

![特征重要性分数图](https://github.com/HunterLC/Features/blob/master/image/feature/feature_importance_0419.png)

### 2020-04-10
1.测试删除了132条相同微博内容的虚假新闻之后的原始数据集（0404_origin_no_dup）效果

2.测试删除了132条相同微博内容的虚假新闻之后的数据集（0404_no_dup）两种特征选择算法的时间

单纯RFE
```
随机森林ACC：0.9475406375343044
随机森林F 1：0.9475207770416302
随机森林AUC：0.9476567196160323
特征个数：108->108
花费时间：436.9608919620514s
```
![0410no_dup F数据集热力图](https://github.com/HunterLC/Features/blob/master/image/feature/heatmap_0404_no_dup_rfe_0410.png)

Filter+RFE
```
随机森林ACC：0.9435296601224403
随机森林F 1：0.9435082811212719
随机森林AUC：0.9436456304829761
特征个数：108->68
花费时间：343.72684717178345s
```
![0410no_dup F+R数据集热力图](https://github.com/HunterLC/Features/blob/master/image/feature/heatmap_0404_no_dup_filter_rfe_0410.png)
### 2020-04-09
1.去除虚假新闻中132列高度相似的微博文本
### 2020-04-06
1.罪过，提取新数据集的时候我忘记了保存未经特征选择的数据集，还好有那个100维w2v完整的数据集，因此可以在它上面删除那100w2v，然后提取新的64维，并按照预处理的删除那445行index，concat起来就是新的结果

2.测试原数据集和特征选择后的时间性能对比

原数据集
```
特征约简前的数据读取时间：36.96369194984436s
特征约简前模型运行时间：119.37581896781921s
特征约简前全程运行时间：156.33951091766357s
随机森林ACC：0.9019669717050595
随机森林F 1：0.901790121612329
随机森林AUC：0.9028155033878457
```
![0406原数据集ROC](https://github.com/HunterLC/Features/blob/master/image/feature/roc_0404_origin_0406.png)

![0406原数据集热力图](https://github.com/HunterLC/Features/blob/master/image/feature/heatmap_0404_origin_0406.png)

Filter+RFE数据集
```
特征约简后的数据读取时间：1.4999802112579346s
特征约简后模型运行时间：8.606698513031006s
特征约简后全程运行时间：10.10667872428894s
随机森林ACC：0.9445671610392342
随机森林F 1：0.9445533051217928
随机森林AUC：0.9449984953041004
```
![0406Filter+RFE数据集ROC](https://github.com/HunterLC/Features/blob/master/image/feature/roc_0404_0406.png)

![0406Filter+RFE数据集热力图](https://github.com/HunterLC/Features/blob/master/image/feature/heatmap_0404_0406.png)

3.特征选择时间对比

单纯使用RFE
```
随机森林ACC：0.9418323340696328
随机森林F 1：0.9418190883997918
随机森林AUC：0.9422521928762498
特征个数：108->88
共计时间：450.77246856689453
```
![0406RFE数据集热力图](https://github.com/HunterLC/Features/blob/master/image/feature/heatmap_0404_rfe_only_0406.png)

Filter+RFE
```
随机森林ACC：0.9403597349321553
随机森林F 1：0.9403502325005774
随机森林AUC：0.9407435349206761
特征个数：108->68
共计时间：340.9696593284607
```
![0406Filter+RFE数据集热力图](https://github.com/HunterLC/Features/blob/master/image/feature/heatmap_0404_filter_rfe_0406.png)

### 2020-04-05
1.尝试xgboost结合SelectFromModel
```
model = XGBClassifier(learning_rate=0.1,
                          n_estimators=1000,  # 树的个数--1000棵树建立xgboost
                          max_depth=6,  # 树的深度
                          min_child_weight=1,  # 叶子节点最小权重
                          gamma=0.,  # 惩罚项中叶子结点个数前的参数
                          subsample=0.8,  # 随机选择80%样本建立决策树
                          colsample_btree=0.8,  # 随机选择80%特征建立决策树
                          random_state=27  # 随机数
                          )
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.000,n=144,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.001,n=118,Accuracy:94.00%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.003,n=100,Accuracy:93.97%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.004,n=76,Accuracy:93.83%
Thresh=0.006,n=59,Accuracy:93.83%
Thresh=0.006,n=59,Accuracy:93.83%
Thresh=0.006,n=59,Accuracy:93.83%
Thresh=0.006,n=59,Accuracy:93.83%
Thresh=0.006,n=59,Accuracy:93.83%
Thresh=0.006,n=59,Accuracy:93.83%
Thresh=0.006,n=59,Accuracy:93.83%
Thresh=0.006,n=59,Accuracy:93.83%
Thresh=0.006,n=59,Accuracy:93.83%
Thresh=0.006,n=59,Accuracy:93.83%
Thresh=0.007,n=49,Accuracy:93.58%
Thresh=0.007,n=49,Accuracy:93.58%
Thresh=0.007,n=49,Accuracy:93.58%
Thresh=0.007,n=49,Accuracy:93.58%
Thresh=0.007,n=49,Accuracy:93.58%
Thresh=0.007,n=49,Accuracy:93.58%
Thresh=0.009,n=43,Accuracy:93.35%
Thresh=0.009,n=43,Accuracy:93.35%
Thresh=0.009,n=43,Accuracy:93.35%
Thresh=0.009,n=43,Accuracy:93.35%
Thresh=0.009,n=43,Accuracy:93.35%
Thresh=0.009,n=43,Accuracy:93.35%
Thresh=0.009,n=43,Accuracy:93.35%
Thresh=0.009,n=43,Accuracy:93.35%
Thresh=0.010,n=35,Accuracy:93.22%
Thresh=0.010,n=35,Accuracy:93.22%
Thresh=0.010,n=35,Accuracy:93.22%
Thresh=0.010,n=35,Accuracy:93.22%
Thresh=0.010,n=35,Accuracy:93.22%
Thresh=0.012,n=30,Accuracy:93.24%
Thresh=0.012,n=30,Accuracy:93.24%
Thresh=0.012,n=30,Accuracy:93.24%
Thresh=0.012,n=30,Accuracy:93.24%
Thresh=0.012,n=30,Accuracy:93.24%
Thresh=0.012,n=30,Accuracy:93.24%
Thresh=0.013,n=24,Accuracy:92.76%
Thresh=0.013,n=24,Accuracy:92.76%
Thresh=0.013,n=24,Accuracy:92.76%
Thresh=0.013,n=24,Accuracy:92.76%
Thresh=0.013,n=24,Accuracy:92.76%
Thresh=0.013,n=24,Accuracy:92.76%
Thresh=0.013,n=24,Accuracy:92.76%
Thresh=0.015,n=17,Accuracy:92.51%
Thresh=0.015,n=17,Accuracy:92.51%
Thresh=0.016,n=15,Accuracy:91.66%
Thresh=0.016,n=15,Accuracy:91.66%
Thresh=0.016,n=15,Accuracy:91.66%
Thresh=0.016,n=15,Accuracy:91.66%
Thresh=0.018,n=11,Accuracy:88.94%
Thresh=0.018,n=11,Accuracy:88.94%
Thresh=0.019,n=9,Accuracy:88.14%
Thresh=0.021,n=8,Accuracy:87.77%
Thresh=0.022,n=7,Accuracy:87.19%
Thresh=0.025,n=6,Accuracy:86.42%
Thresh=0.025,n=6,Accuracy:86.42%
Thresh=0.026,n=4,Accuracy:81.26%
Thresh=0.031,n=3,Accuracy:72.60%
Thresh=0.063,n=2,Accuracy:72.64%
Thresh=0.079,n=1,Accuracy:70.64%
```
### 2020-04-04
1.完成新测试集特征的提取

2.对新数据集重新提取情感繁分数，删除无关列
```
# 删除无关列
start = time.time()
# 原始数据读入
df_image = pd.read_csv(test_csv_path)
df_image.drop(['piclist', 'text', 'tf_vgg19_class', 'tf_resnet50_class'], axis=1,inplace=True)
df_image.to_csv(test_csv_path, index=0)  # 不保留行索引
end = time.time()
logging.info("运行时间：" + str(end - start))
```
3.对新数据集进行PCA处理颜色矩和resnet50列特征,经过1 2 3处理后命名为 result_origin_ready.csv
4.对原始数据集进行word2vec64维缩减

### 2020-04-02
1.按照会议内容调整热力图细节，放大矩阵数字大小，横纵坐标标签清晰化
```
def draw_confusion_matrix_heat_map(y_test, rf_pred):
    # 构建混淆矩阵
    cm = pd.crosstab(rf_pred, y_test)
    # 将混淆矩阵构造成数据框，并加上字段名和行名称，用于行和列的含义说明
    cm.columns = ['真实新闻','虚假新闻']
    cm.index = ['真实新闻','虚假新闻']
    sns.set(font_scale=1.5)
    # plt.rc('font', family='Times New Roman', size=12)
    # 绘制热力图
    sns.heatmap(cm, annot=True, cmap='GnBu', fmt='d', annot_kws={'size':25})
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # 添加x轴和y轴的标签
    plt.xlabel('实际标签')
    plt.ylabel('预测标签')
    plt.show()
```
Filter+RFE处理后的数据集性能

![优化热力图细节](https://github.com/HunterLC/Features/blob/master/image/feature/heatmap_0402.png)

同时测试了原数据集的效果

![优化热力图细节](https://github.com/HunterLC/Features/blob/master/image/feature/heatmap_0402_origin.png)
>随机森林ACC：0.9196381613547913
>
>随机森林F 1：0.9193688844569208
>
>随机森林AUC：0.9207483442811507
>
### 2020-03-30
1.开启新测试数据集的提取
先给出特征列表对照

 特征名称              | 意义                 | 数据类型   | 备注                                                                                    |
| ------------------- |:--------------------:| --------:|:---------------------------------------------------------------------------------------- |
|+ id                  | 微博账号标识          |   object  | 无                                                                                       |
|+ text                | 微博正文              |   object  | 无                                                                                       |
|+ text_length         | 微博正文长度          |    int    | 无                                                                                      |
|+ contains_questmark  | 微博正文是否包含 ？    |    int    | 1：包含，0：不包含                                                                        |
|+ num_questmarks      | ？ 的个数             |    int   | 无                                                                                       |
|+ contains_exclammark | 微博正文是否包含 ！    |    int   | 1：包含，0：不包含                                                                        |
|+ num_exclammarks     | ！ 的个数             |    int   | 无                                                                                      |
|+ contains_hashtag    | 微博正文是否包含话题   |    int   | 1：包含，0：不包含                                                                        |
|+ num_hashtags        | 话题个数              |    int   | 新浪微博话题格式为两个#，例如 #春节#                                                       |
|+ contains_URL        | 微博正文是否包含链接   |    int   | 1：包含，0：不包含                                                                        |
|+ num_URLs            | 链接个数              |    int   | 链接需包含http头的，例如http://www.baidu.com 计数1；不包含http头的例如www.baidu.com不计数     |
|+ contains_mention    | 微博正文是否包含提及@  |    int   | 1：包含，0：不包含                                                                        |
|+ num_mentions        | @ 的个数             |    int    | 无                                                                                      |
|+ sentiment_score     | 情感分数             |    float  | 使用snownlp打分，取值0~1之间，越接近0越消极，越接近1越积极                                   |
|+ num_noun            | 名词个数             |    int    | 无                                                                                      |
|+ num_verb            | 动词个数             |    int    | 无                                                                                      |
|+ num_pronoun         | 代词个数             |    int    | 无                                                                                      |
|+ word2vec_1~64      | 词向量列1~64        |    float  | 采用word2vec构建词向量，每个正文利用jieba分词后计算词矩阵之和，并取平均值作为文本的词向量       |
|+ category            | 微博新闻类属         |    object | 无                                                                                      |
|+ label               | 真假新闻标签         |    int    | 0：真新闻 ，1:假新闻                                                                      |
|+ num_possentiwords   | 积极词汇个数         |    int    | 无                                                                                      |
|+ num_negsentiwords   | 消极词汇个数         |    int    | 无                                                                                      |
|+ contains_firstorderpron | 是否包含第一人称  |   int     | 1:有，0：无          |
|+ contains_secondorderpron| 是否包含第二人称  |   int     | 1:有，0：无          |
|+ contains_thirdorderpron | 是否包含第三人称  |   int     | 1:有，0：无          |
|+ user_gender       | 用户性别        |    object   | 男、女     |
|+ user_follow_count | 用户关注数      |    float    | 无         |
|+ user_fans_count   | 用户粉丝数      |    float    | 无         |
|+ user_weibo_count  | 用户微博数      |    float    | 无         |
|+ folfans_ratio     | 关注/粉丝比     |    float    | 无         |
|+ user_location     | 用户所在地      |    float    | 无         |
|+ user_description  | 用户个性签名    |    float    | 无         |
|+ h_first_moment    | 色相一阶矩      |   float   | 无          |
|+ s_first_moment    | 饱和度一阶矩    |    float   | 无          |
|+ v_first_moment    | 亮度一阶矩      |    float    | 无         |
|+ h_second_moment   | 色相二阶矩      |    float    | 无         |
|+ s_second_moment   | 饱和度二阶矩    |    float    | 无         |
|+ v_second_moment   | 亮度二阶矩      |    float    | 无         |
|+ h_third_moment    | 色相三阶矩      |    float    | 无         |
|+ s_third_moment    | 饱和度三阶矩    |    float    | 无         |
|+ v_third_moment    | 亮度三阶矩      |    float    | 无         |
|+ resnet_1~2048    | resnet50特征      |   float   | 无          |
|+ sim_image_word  | 图文相似度  |   float        |采用词嵌入方式，减少中英翻译误差中的影响|
|+ tf_vgg19_class | vgg19分类     |   object     | 无          |
|+ tf_resnet_class| resnet50分类  |   object     | 无          |
|+ image_width    | 图片宽度      |   int     | 无          |
|+ image_height   | 图片高度      |   int     | 无          |
|+ image_kb       | 图片物理大小  |   float   | 单位为kb     |

### 2020-03-28
1.统计filter+rfe进行特征选择的时间和单纯只用rfe的时间
>filter+rfe
>
>随机森林ACC：0.9555064689176397
>
>随机森林F 1：0.9554910338022102
>
>随机森林AUC：0.9559837050155026
>
>filter+RFE进行特征选择的时间：734.2702012062073
>
>rfe
>
>随机森林ACC：0.9566635110970864
>
>随机森林F 1：0.9566459706584156
>
>随机森林AUC：0.9571642427590218
>
>单纯RFE进行特征选择的时间：1425.6536462306976s
### 2020-03-25
1.添加了所训练的sklearn模型的保存和加载

### 2020-03-24
1.添加了特征重要性排行

![特征重要排行](https://github.com/HunterLC/Features/blob/master/image/feature/Figure_importance.png)

### 2020-03-21
1.本来想尝试新的测试集，但我随后发现真实事件的数据集缺少部分十分重要的特征信息，比如用户的微博、粉丝数等等，这种数据不能够自己编造或者提取到。

2.测试了一下rfe的特征个数和性能，绘制了如下图：

![特征个数及性能](https://github.com/HunterLC/Features/blob/master/image/feature/rfe_selected_feature.png)
### 2020-03-20
1.添加Filter特征选择算法之单变量特征选择，快速进行特征筛选，再利用RFE进行选择，最终特征降至90,就我目测而言，mutual_info_classif效果一般
```
# 可选项f_classif, chi2, mutual_info_classif
SelectKBest(mutual_info_classif, k=90)
```
>随机森林ACC：
 0.954033869780162
>
>随机森林F 1：
 0.9540141965244037
>
>随机森林AUC：
 0.954542756161971

 2.对新数据集进行提取和完善，用做测试集
 
### 2020-03-19
1.添加Filter特征选择算法之单变量特征选择，快速进行特征筛选，再利用RFE进行选择，最终特征降至70
```
# 可选项f_classif, chi2, mutual_info_classif
SelectKBest(f_classif, k=90)
```

>随机森林ACC：
 0.9551909119596087
>
>随机森林F 1：
 0.9551751739577907
>
>随机森林AUC：
 0.9556698393512464
>
### 2020-03-13
1.经过商量，决定先把整个特征工程的流程先走通，KPCA降维在后面返校后进行实验补充

2.添加时间性能，对比原数据和特征选择后的数据的训练时间

特征约简前
>特征约简前的数据读取时间：36.12473177909851s
>
>特征约简前模型运行时间：141.8570680618286s
>
>特征约简前全程运行时间：177.98179984092712s
>
>随机森林ACC：
 0.9218470600610077
>
>随机森林F 1：
 0.9215803336050783
>
>随机森林AUC：
 0.9229667857526413
>
特征约简后
>特征约简后的数据读取时间：1.0653886795043945s
>
>特征约简后模型运行时间：7.845198631286621s
>
>特征约简后全程运行时间：8.910587310791016s
>
>随机森林ACC：
 0.9573998106658251
>
>随机森林F 1：
 0.9573835348663006
>
>随机森林AUC：
 0.9578918444596867
### 2020-03-10
1.经过实践，在我的电脑上跑不了KPCA代码，MemoryError。同时KCCA教程过少，不能调用库来进行测试。

2.目前该模型效果过拟合了，特别是在于图片存在重复。关于测试集，目前手头上准备了两个，但是缺少label，目前还不能用于测试。一旦label确定之后，我还需要重新处理我的测试集，不过好处在于经过特征选择之后我的目标特征就少很多了，在提取的时候还可比较省事。

### 2020-03-09
1.在数据集层面将所有的object列转为数字编码，避免独热编码时出现冗余列。
  RFE，step=10,cv=10,将特征从144列降至94列，效果提升了 0.957 -> 0.958
```
    gender_map   = {'男': 1, '女': 0}

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
```
>随机森林ACC：
 0.9582412958872409
>
>随机森林F 1：
 0.9582257141172548
>
>随机森林AUC：
 0.9587300074433529
>
### 2020-03-06
1.采用RFE，step=10,目前特征降至128维，速度更快，准确率在0.957左右
>随机森林ACC：
 0.9571894393604712
>
>随机森林F 1：
 0.9571682040998659
>
>随机森林AUC：
 0.9577241764479276
>
2.存在问题，用户地址和用户性别之类的object在运算过程中转为独热码，特征列会被删除，同时效果感觉一般
### 2020-03-05
1.采用pca（主成分分析法），将imagenet50特征从2048维降至10，将Color Moment从9降至2，删除特征列'tf_vgg19_class','tf_resnet50_class'时
>[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 4 concurrent workers.
>
>[Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    3.4s
>
>[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    7.9s finished
>
>[Parallel(n_jobs=4)]: Using backend ThreadingBackend with 4 concurrent workers.
>
>[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:    0.0s
>
>[Parallel(n_jobs=4)]: Done 100 out of 100 | elapsed:    0.0s finished
>
>随机森林ACC：
 0.9573998106658251
>
>随机森林F 1：
 0.9573829561575022
>
>随机森林AUC：
> 0.9580041019681674
>
准确度从0.93提高至0.95

![ROC_AUC](https://github.com/HunterLC/Features/blob/master/image/feature/roc_0305.png)

![heatmap](https://github.com/HunterLC/Features/blob/master/image/feature/heatmap_0305.png)

### 2020-03-04
以目前的数据集为主：

1.新增分类指标AUC，目前分类指标有三个：ACC、AUC、F1
>[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 4 concurrent workers.
>
>[Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    8.1s
>
>[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   18.9s finished
>
>[Parallel(n_jobs=4)]: Using backend ThreadingBackend with 4 concurrent workers.
>
>[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:    0.0s
>
>[Parallel(n_jobs=4)]: Done 100 out of 100 | elapsed:    0.0s finished
>
>随机森林ACC：
> 0.9392026927527085
>
>随机森林F 1：
> 0.9390816421867911
>
>随机森林AUC：
> 0.940097542719596
>
2.新增ROC曲线绘制，用于直观展示AUC值

![ROC_AUC](https://github.com/HunterLC/Features/blob/master/image/feature/roc_0304.png)

3.新增分类结果的混淆矩阵热力图绘制

![heatmap](https://github.com/HunterLC/Features/blob/master/image/feature/heatmap_0304.png)

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
