import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

fusion_csv_path = r'G:\毕设\数据集\微博\fusion_news_features.csv'

def knn_classifier(X_train, y_train):
    '''
    k最近邻分类
    '''
    model_knn = KNeighborsClassifier(n_neighbors = 2, weights = 'distance')
    model_knn.fit(X_train, y_train)
    return model_knn

def classifier(data_file):
    '''
    data_file : CSV文件
    '''
    df = pd.read_csv(data_file)
    df.drop('id', axis=1, inplace=True)  # 删除列（axis=1指定，默认为行），并将原数据置换为新数据（inplace=True指定，默认为False）

    feature_attr = [i for i in df.columns if i not in ['label']]
    label_attr = 'label'
    df.fillna(0, inplace=True)
    df = df[17200:21200]
    # 特征预处理
    obj_attrs = []
    for attr in feature_attr:
        if df.dtypes[attr] == np.dtype(object):  # 添加离散数据列
            obj_attrs.append(attr)
    if len(obj_attrs) > 0:
        df = pd.get_dummies(df, columns=obj_attrs)  # 转为哑变量

    y = df[label_attr].astype('category').cat.codes.values  # 将label_attr列转换为分类/分组类型
    df.drop(label_attr, axis=1, inplace=True)  # 删除列（axis=1指定，默认为行），并将原数据置换为新数据（inplace=True指定，默认为False）
    X = df.values

    # 采用10折交叉验证
    kf = KFold(n_splits=10)


    knn_hist = []
    for train_index, test_index in kf.split(X):
        # 加载数据
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 训练
        knn_model = knn_classifier(X_train, y_train)

        # 预测
        knn_pred = knn_model.predict(X_test)
        print('K最近邻模型的准确率：\n', metrics.accuracy_score(y_test, knn_pred))
        print('K最近邻模型的评估报告：\n', metrics.classification_report(y_test, knn_pred))

        # 评估阶段：
        #    F度量又称F1分数或F分数
        #    F1= 2 * ( precision * recall ) / ( precison + recall )
        knn_f1 = f1_score(y_test, knn_pred, average='micro')

        # 结果汇总
        knn_hist.append(knn_f1)

    # 绘图

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(18, 10), gridspec_kw={'width_ratios': [5, 3]})
    # 绘制每轮计算的F1值折线图
    ax0.set_title('每轮F1值', color='black')
    ax0.plot(knn_hist, linestyle=':', color='black', label='k最近邻')
    # 添加图例
    ax0.legend()

    # 绘制平均F1值直方图
    ax1.set_title('平均F1值', color='black')
    x_names = ['k最近邻']
    y_data = [np.mean(knn_hist),]
    # print(np.mean(dt_hist), np.mean(knn_hist), np.mean(nb_hist))
    ax1.bar(x=np.arange(len(x_names)), height=y_data)
    # 添加直方图数据
    for x, y in enumerate(y_data):
        plt.text(x, y + 0.01, '%.2f' % y, ha='center')
    # 添加刻度标签
    plt.xticks(np.arange(len(x_names)), x_names, fontsize=12)
    plt.show()

classifier(fusion_csv_path)