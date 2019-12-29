import pandas as pd
import numpy as np

#数据集读入
news_data = pd.read_csv(r'G:\毕设\数据集\微博\train.csv')
#查看数据集是否存在缺失值
# id                     0
# text                   0
# piclist            16639
# userGender           445
# userFollowCount      464
# userFansCount        448
# userWeiboCount       472
# userLocation         537
# userDescription     7715
# category               0
# label                  0
news_data.apply(lambda x:np.sum(x.isnull()))
#查看连续型数据的描述性统计值  userFollowCount  userFansCount  userWeiboCount label  4
news_data.describe()
#查看离散型数据的描述性统计值
news_data.describe(include=['object'])
#查看所有数据 38471 rows x 11 columns
news_data.all