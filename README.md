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
| num_URLs            | 链接个数              |    int   | 链接需包含http头的，例如http://www.baidu.com计数1； 不包含http头的例如www.baidu.com不计数     |
| contains_mention    | 微博正文是否包含提及@  |    int   | 1：包含，0：不包含                                                                        |
| num_mentions        | @ 的个数             |    int    | 无                                                                                      |
| sentiment_score     | 情感分数             |    float  | 使用snownlp打分，取值0~1之间，越接近0越消极，越接近1越积极                                   |
| num_noun            | 名词个数             |    int    | 无                                                                                      |
| num_verb            | 动词个数             |    int    | 无                                                                                      |
| num_pronoun         | 代词个数             |    int    | 无                                                                                      |
| word2vec_1~100      | 词向量列1~100        |    float  | 采用word2vec构建词向量，每个正文利用jieba分词后计算词矩阵之和，并取平均值作为文本的词向量       |
| category            | 微博新闻类属         |    object | 无                                                                                      |
| label               | 真假新闻标签         |    int    | 0：真新闻 ，1:假新闻                                                                      |

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