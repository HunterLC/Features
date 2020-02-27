import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, model_selection
import matplotlib.pyplot as plt

fusion_csv_path = r'G:\毕设\数据集\微博\fusion_news_features.csv'
text_csv_path = r'G:\毕设\数据集\微博\text.csv'
user_csv_path = r'G:\毕设\数据集\微博\user.csv'
image_csv_path = r'G:\毕设\数据集\微博\image.csv'


def get_save_index():
    df_user = pd.read_csv(user_csv_path, usecols=['user_gender'])
    # 保留user_gender列中的非空行，非空为True，空行为False
    save_index = df_user.isnull().sum(axis=1) == 0
    return save_index

def features_preprocessor(df):
    #获取需要保留的index行
    save_index = get_save_index()
    #剔除 455 行用户特征缺少严重的行
    df = df[save_index]
    #文本数据预处理

    #用户数据预处理
    #图片数据预处理
    return df

def knn_classifier(X_train, y_train):
    '''
    k最近邻分类
    '''
    # model_knn = KNeighborsClassifier(n_neighbors = 6, weights = 'distance')
    # model_knn = LinearSVC(C=0.1)
    # model_knn = decision_tree_classifier(X_train, y_train)
    model_knn = RandomForestClassifier(n_estimators=100,
                                   bootstrap=True,
                                   max_features='sqrt')
    model_knn.fit(X_train, y_train)
    return model_knn

def decision_tree_classifier(X_train, y_train):
    '''
    决策树分类
    '''
    # params = dt_search_best(X_train, y_train)
    # print(params['max_depth'],params['min_samples_leaf'],params['min_samples_split'])
    # model_dt = DecisionTreeClassifier(max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'],min_samples_split=params['min_samples_split'])
    model_dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=4,
                                      min_samples_split=6)
    model_dt.fit(X_train, y_train)
    return model_dt

def dt_search_best(X_train, y_train):
    '''
    采用网格搜索法确定决策树最佳组合参数值
    '''
    #预设各参数的不同选项值
    max_depth = [2, 3, 4, 5, 6]
    min_samples_split = [2, 4, 6, 8]
    min_samples_leaf = [2, 4, 8, 10, 12]
    #将各参数值以字典形式组织起来
    parameters = {'max_depth': max_depth, 'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf}
    # 网格搜索法,测试不同的参数值
    grid_dtcateg= GridSearchCV(estimator = DecisionTreeClassifier(),
                               param_grid = parameters, cv = 10)
    #模型拟合
    grid_dtcateg.fit(X_train, y_train)
    #返回最佳组合的参数值
    print(grid_dtcateg.best_params_)
    return grid_dtcateg.best_params_

def rf_classifier(data_file, label='label'):
    df = pd.read_csv(data_file)
    df_user = pd.read_csv(user_csv_path).drop(columns = ['id'], axis = 1)
    df = pd.concat([df, df_user], axis=1)
    df.drop(['id'], axis=1, inplace=True)  # 删除列（axis=1指定，默认为行），并将原数据置换为新数据（inplace=True指定，默认为False）
    df = features_preprocessor(df)
    feature_attr = [i for i in df.columns if i not in [label]]
    label_attr = label
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
    model_rf = RandomForestClassifier(n_estimators=100,
                                       bootstrap=True,
                                       max_features='sqrt')
    model_rf.fit(X_train, y_train)
    rf_pred = model_rf.predict(X_test)
    print('随机森林准确率：\n', metrics.accuracy_score(y_test, rf_pred))


def classifier(data_file):
    '''
    data_file : CSV文件
    '''
    df = pd.read_csv(data_file)
    df_user = pd.read_csv(user_csv_path).drop(columns=['id'], axis=1)
    df_image = pd.read_csv(image_csv_path).drop(columns=['piclist','tf_vgg19_class','tf_resnet50_class'], axis=1)
    df = pd.concat([df['label'], df_user, df_image], axis=1)
    df.drop(['id'], axis=1, inplace=True)  # 删除列（axis=1指定，默认为行），并将原数据置换为新数据（inplace=True指定，默认为False）
    df = features_preprocessor(df)
    feature_attr = [i for i in df.columns if i not in ['label']]
    label_attr = 'label'
    df.fillna(0, inplace=True)
    df = df[15200:23200]
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
    feature_importances = knn_model.feature_importances_
    features_list = df.columns.values.tolist()
    sorted_idx = np.argsort(feature_importances)

    # 绘图

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(18, 10), gridspec_kw={'width_ratios': [5, 3]})
    # 绘制每轮计算的F1值折线图
    ax0.set_title('每轮F1值', color='black')
    ax0.plot(knn_hist, linestyle=':', color='black', label='随机森林')
    # 添加图例
    ax0.legend()

    # 绘制平均F1值直方图
    ax1.set_title('平均F1值', color='black')
    x_names = ['随机森林']
    y_data = [np.mean(knn_hist),]
    # print(np.mean(dt_hist), np.mean(knn_hist), np.mean(nb_hist))
    ax1.bar(x=np.arange(len(x_names)), height=y_data)
    # 添加直方图数据
    for x, y in enumerate(y_data):
        plt.text(x, y + 0.01, '%.2f' % y, ha='center')
    # 添加刻度标签
    plt.xticks(np.arange(len(x_names)), x_names, fontsize=12)
    plt.show()

    # plt.figure(figsize=(5, 7))
    # plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    # plt.yticks(range(len(sorted_idx)), np.array(features_list)[sorted_idx])
    # plt.xlabel('Importance')
    # plt.title('Feature importances')
    # plt.draw()
    # plt.show()

classifier(text_csv_path)
# df_text = pd.read_csv(text_csv_path)
# df_user = pd.read_csv(user_csv_path,usecols='user_gender')
# # 保留user_gender列中的非空行，非空为True，空行为False
# save_index = df_user.isnull().sum(axis=1) == 0
# num = 0
# for a in save_index:
#     if(a==True):
#         num += 1
# print(num)
# df_user_new = df_user[save_index]
# df_user.dropna(how='all',axis=0,inplace=True)
# df_user.apply(lambda x:np.sum(x.isnull()))
# df_text.drop(index=drop_index[1],inplace=True)