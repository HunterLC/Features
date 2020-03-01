from sklearn.feature_selection import RFECV
import pandas as pd
import numpy as np
from sklearn import metrics, model_selection
from sklearn.ensemble import RandomForestClassifier

fusion_csv_path = r'G:\毕设\数据集\微博\fusion_news_features.csv'
text_csv_path = r'G:\毕设\数据集\微博\text.csv'
user_csv_path = r'G:\毕设\数据集\微博\user.csv'
image_csv_path = r'G:\毕设\数据集\微博\image.csv'

def rf_classifier(data_file, label='label'):
    df = pd.read_csv(data_file)
    df.drop(['id'], axis=1, inplace=True)  # 删除列（axis=1指定，默认为行），并将原数据置换为新数据（inplace=True指定，默认为False）
    feature_attr = [i for i in df.columns if i not in [label]]
    label_attr = label
    df['user_description'] = df['user_description'].astype(float)
    df.fillna(0, inplace=True)
    # 特征预处理
    obj_attrs = []
    for attr in feature_attr:
        if df.dtypes[attr] == np.dtype(object):  # 添加离散数据列
            obj_attrs.append(attr)
    if len(obj_attrs) > 0:
        df = pd.get_dummies(df, columns=obj_attrs)  # 转为哑变量
    # df = df[15200:23200]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(df.drop(label_attr, axis=1),
                                                                        df.label,
                                                                        test_size=0.25,
                                                                        random_state=1234)
    estimator = RandomForestClassifier(n_estimators=100,
                                       bootstrap=True,
                                       max_features='sqrt')

    '''
    estimator：该参数传入用于递归构建模型的有监督型基学习器，要求该基学习器具有fit方法，且其输出含有coef_或feature_importances_这种结果；

　　step：数值型，默认为1，控制每次迭代过程中删去的特征个数，有以下两种情况：

　　　　1.若传入大于等于1的整数，则在每次迭代构建模型的过程中删去对应数量的特征；

　　　　2.若传入介于0.0到1.0之间的浮点数，则在每次第迭代构造模型的过程中删去对应比例的特征。

　　cv：控制交叉验证的分割策略，默认是3折交叉验证，有以下几种情况：

　　　　1.None，等价于不传入参数，即使用默认设置的3折交叉验证；

　　　　2.正整数，这时即指定了交叉验证中分裂的子集个数，即k折中的k；

　　n_jobs：控制并行运算中利用到的CPU核心数，默认为1，即单核工作，若设置为-1，则启用所有核心进行运算；

　　函数返回值：　

　　n_features_：通过交叉验证过程最终剩下的特征个数；

　　support_：被选择的特征的被选择情况（True表示被选择，False表示被淘汰）；

　　ranking_：所有特征的评分排名；

　　estimator_：利用剩下的特征训练出的模型；
    '''
    rfe_model_rf = RFECV(estimator, step=1, cv=10, scoring=None, verbose=0, n_jobs=-1)
    rfe_model_rf = rfe_model_rf.fit(X_train, y_train)
    rf_pred = rfe_model_rf.predict(X_test)
    print('随机森林准确率：\n', metrics.accuracy_score(y_test, rf_pred))
    return rfe_model_rf

model = rf_classifier(fusion_csv_path)