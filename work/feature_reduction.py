from sklearn.feature_selection import RFECV
from sklearn.decomposition import KernelPCA
import pandas as pd
import numpy as np
from sklearn import metrics, model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy import exp
import time
from numpy.linalg import eigh

fusion_csv_path = r'G:\毕设\数据集\微博\fusion_news_features.csv'
new_fusion_csv_path = r'G:\毕设\数据集\微博\fusion_features_0306.csv'
new_0312_fusion_csv_path = r'G:\毕设\数据集\微博\fusion_features_0312.csv'
fusion_no_object_csv_path = r'G:\毕设\数据集\微博\fusion_features_0306_no_object.csv'
text_csv_path = r'G:\毕设\数据集\微博\text.csv'
user_csv_path = r'G:\毕设\数据集\微博\user.csv'
image_csv_path = r'G:\毕设\数据集\微博\image.csv'
selected_features_data_path = r'data_selection.txt'


def decision_tree_classifier(X_train, y_train):
    '''
    决策树分类
    '''
    # params = dt_search_best(X_train, y_train)
    # print(params['max_depth'],params['min_samples_leaf'],params['min_samples_split'])
    # model_dt = DecisionTreeClassifier(max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'],min_samples_split=params['min_samples_split'])

    return


def rf_search_best(X_train, y_train):
    '''
    采用网格搜索法确定随机森林最佳组合参数值
    '''
    # 预设各参数的不同选项值
    max_depth = [18, 19, 20, 21, 22]  # 数据量小在10左右，数据量大在20左右
    min_samples_split = [2, 4, 6, 8]
    min_samples_leaf = [2, 4, 8, 10, 12]
    # 将各参数值以字典形式组织起来
    parameters = {'max_depth': max_depth, 'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf}
    # 网格搜索法,测试不同的参数值
    grid_dtcateg = GridSearchCV(estimator=RandomForestClassifier(),
                                param_grid=parameters, cv=10)
    # 模型拟合
    grid_dtcateg.fit(X_train, y_train)
    # 返回最佳组合的参数值
    print(grid_dtcateg.best_params_)
    return grid_dtcateg.best_params_


def read_data_frame(data_file, use_cols=None, drop_cols=None):
    df = pd.read_csv(data_file, usecols=use_cols)
    if drop_cols != None:
        df.drop(drop_cols, axis=1, inplace=True)
    return df


def rf_classifier(df, label='label'):
    # 删除列（axis=1指定，默认为行），并将原数据置换为新数据（inplace=True指定，默认为False）
    # df.drop(['id', 'tf_vgg19_class', 'tf_resnet50_class'], axis=1,inplace=True)
    # df_1 = pd.read_csv('colorf.csv', names=['pca_color_moment1', 'pca_color_moment2'])
    # df_2 = pd.read_csv('10_resnet.csv', names=['pca_net1', 'pca_net2', 'pca_net3', 'pca_net4', 'pca_net5',
    #                                            'pca_net6', 'pca_net7', 'pca_net8', 'pca_net9', 'pca_net10'])
    # df = pd.concat([df, df_1, df_2], axis=1)
    # df.to_csv(new_fusion_csv_path, index=0)  # 不保留行索引
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

    X_train, X_test, y_train, y_test = model_selection.train_test_split(df.drop(label, axis=1),
                                                                        df['label'],
                                                                        test_size=0.25,
                                                                        random_state=1234)
    # 构造随机森林的分类器
    estimator = RandomForestClassifier(max_depth=20,
                                       min_samples_leaf=4,
                                       min_samples_split=6,
                                       n_estimators=100,
                                       bootstrap=True,
                                       max_features='sqrt',
                                       verbose=1,
                                       n_jobs=-1)
    # RFE递归特征消除算法进行特征选择
    # rfe_model_rf = selection_rfe(estimator, X_train, y_train)
    estimator = estimator.fit(X_train, y_train)
    rf_pred = estimator.predict(X_test)
    print('随机森林ACC：\n', metrics.accuracy_score(y_test, rf_pred))
    print('随机森林F 1：\n', metrics.f1_score(y_test, rf_pred, average='weighted'))
    print('随机森林AUC：\n', metrics.roc_auc_score(y_test, rf_pred))
    # 绘制ROC曲线，一般认为AUC大于0.8即算较好效果
    draw_auc(estimator, X_test, y_test)
    # 绘制混淆矩阵热力图
    draw_confusion_matrix_heat_map(y_test, rf_pred)
    return df, estimator


def extraction_pca(df, count=2):
    pca = PCA(n_components=count)  # mle代表自动选择最后保存的特征数量
    pca.fit(df)
    # 降维
    low_dimensionality = pca.transform(df)
    return pca, pd.DataFrame(low_dimensionality)


def extraction_kpca(df):
    kpca = KernelPCA(n_components=10, kernel='poly', degree=3, gamma=5, n_jobs=-1)
    df = df[13200:33200]
    kpca.fit(df)
    X_skernpca = kpca.transform(df)
    return kpca, pd.DataFrame(X_skernpca)


def draw_auc(estimator, X_test, y_test):
    # 计算绘图数据
    y_score = estimator.predict_proba(X_test)[:, 1]
    # roc_curve函数的第二个参数代表正例的预测概率，而不是实际的预测值
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    # 绘图
    plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
    plt.plot(fpr, tpr, color='black', lw=1)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.text(0.5, 0.3, 'ROC Curve (area = %0.2f)' % roc_auc)
    plt.xlabel('specificity')
    plt.ylabel('sensitivity')
    plt.show()


def selection_rfe(estimator, X_train, y_train):
    """
    Wrapper类特征选择方法——RFE
    :param estimator: 监督型基学习器
    :param X_train: 训练集数据
    :param y_train: 训练集分类数据
    :return: 分类器模型
    """
    rfe_model_rf = RFECV(estimator, step=10, cv=10, scoring=None, verbose=1, n_jobs=-1)
    '''
        estimator：该参数传入用于递归构建模型的有监督型基学习器，要求该基学习器具有fit方法，且其输出含有coef_或feature_importances_这种结果；

    　　step：数值型，默认为1，控制每次迭代过程中删去的特征个数，有以下两种情况：

    　　　　1.若传入大于等于1的整数，则在每次迭代构建模型的过程中删去对应数量的特征；

    　　　　2.若传入介于0.0到1.0之间的浮点数，则在每次第迭代构造模型的过程中删去对应比例的特征。

    　　cv：控制交叉验证的分割策略，默认是3折交叉验证，有以下几种情况：

    　　　　1.None，等价于不传入参数，即使用默认设置的3折交叉验证；

    　　　　2.正整数，这时即指定了交叉验证中分裂的子集个数，即k折中的k；

       verbose:指定计算过程中是否生成日志信息，默认为0，不输出

    　　n_jobs：控制并行运算中利用到的CPU核心数，默认为1，即单核工作，若设置为-1，则启用所有核心进行运算；

    　　函数返回值：　

    　　n_features_：通过交叉验证过程最终剩下的特征个数；

    　　support_：被选择的特征的被选择情况（True表示被选择，False表示被淘汰）；

    　　ranking_：所有特征的评分排名；

    　　estimator_：利用剩下的特征训练出的模型；
        '''
    rfe_model_rf = rfe_model_rf.fit(X_train, y_train)
    return rfe_model_rf


def draw_confusion_matrix_heat_map(y_test, rf_pred):
    # 构建混淆矩阵
    cm = pd.crosstab(rf_pred, y_test)
    # 将混淆矩阵构造成数据框，并加上字段名和行名称，用于行和列的含义说明
    # cm = pd.DataFrame(cm, columns=['fake', 'true'], index=['fake', 'true'])
    # 绘制热力图
    sns.heatmap(cm, annot=True, cmap='GnBu', fmt='d')
    # 添加x轴和y轴的标签
    plt.xlabel('Real Label')
    plt.ylabel('Predict Label')
    plt.show()


def extraction_rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]

    gamma: float
      Tuning parameter of the RBF kernel

    n_components: int
      Number of principal components to return

    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
       Projected dataset

    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.linalg.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i]
                            for i in range(1, n_components + 1)))

    return X_pc


def save_selected_features(df, estimator, save_path=selected_features_data_path):
    """
    保存特征选择后的特征列表
    :param df: dataframe
    :param estimator: 分类器
    :return: 保存文件到指定位置
    """
    i = 0
    j = 0
    features_list = []
    df_list = [x for x in df.columns if x not in ['label']]
    for item in df_list:
        try:
            if estimator.support_[i]:
                features_list.append(item + '\n')
                j += 1
        except:
            print('error')
        i += 1
    print(str(j))
    with open(save_path, 'w+') as f:
        f.writelines(features_list)


def get_selected_features(path=selected_features_data_path):
    """
    加载特征选择后保留的特征列表
    :return:
    """
    my_words = []
    f = open(path, "r", encoding='UTF-8')
    for eachWord in f.readlines():
        my_words.append(eachWord.strip())
    f.close()
    return my_words


def feature_pca(data_path=fusion_csv_path, pca_list=None):
    # pca处理
    if pca_list != None:
        # 处理颜色矩color moment的9列
        df_cm = read_data_frame(data_path, use_cols=pca_list[:9])
        pca_cm, df_new_cm = extraction_pca(df_cm, count=2)
        df_new_cm.columns = ['pca_color_moment1','pca_color_moment2']
        print(list(pca_cm.explained_variance_ratio_)) # pca转换数据的可信率列表
        print(pca_cm.n_components_)    # pca保留的特征列数
        # 处理resnet的2048列
        df_rn = read_data_frame(data_path, use_cols=pca_list[9:])
        pca_rn, df_new_rn = extraction_pca(df_rn, count=10)
        df_new_rn.columns = ['pca_net1', 'pca_net2', 'pca_net3', 'pca_net4', 'pca_net5',
                             'pca_net6', 'pca_net7', 'pca_net8', 'pca_net9', 'pca_net10']

        df = read_data_frame(data_path, drop_cols=pca_list)
        # 特征拼接
        df_pca = pd.concat([df, df_new_cm, df_new_rn], axis=1)
        return df_pca


# pca处理需要加载的列
# pca_list = ['h_first_moment', 's_first_moment', 'v_first_moment',
#             'h_second_moment', 's_second_moment', 'v_second_moment',
#             'h_third_moment', 's_third_moment', 'v_third_moment']
# for i in range(1, 2049):
#     pca_list.append('resnet_' + str(i))
# df = feature_pca(fusion_csv_path, pca_list=pca_list)

#原始数据
# original_start_time = time.time()  # 开始计时
# df_original = pd.read_csv(fusion_csv_path)
# original_read_time = time.time()
# print('特征约简前的数据读取时间：' + str(original_read_time - original_start_time) + 's')
# df_original, estimator_original = rf_classifier(df_original)
# original_end_time = time.time()  #训练结束时间
# print('特征约简前模型运行时间：' + str(original_end_time - original_read_time) + 's')
# print('特征约简前全程运行时间：' + str(original_end_time - original_start_time) + 's')


#获得 pca+特征选择 后的特征子集
selected_features = get_selected_features()
selected_features.append('label')
reduction_start_time = time.time()  # 开始计时
df_reduction = read_data_frame(fusion_no_object_csv_path, use_cols=selected_features)
reduction_read_time = time.time()
print('特征约简后的数据读取时间：' + str(reduction_read_time - reduction_start_time) + 's')
df_reduction, estimator_reduction = rf_classifier(df_reduction)
reduction_end_time = time.time()  #训练结束时间
print('特征约简后模型运行时间：' + str(reduction_end_time - reduction_read_time) + 's')
print('特征约简后全程运行时间：' + str(reduction_end_time - reduction_start_time) + 's')

#测试KPCA降维word2vec的代码段
# use_list = ['text_length',
#             'contains_exclammark',
#             'num_exclammarks',
#             'num_hashtags',
#             'num_noun',
#             'category',
#             'user_fans_count',
#             'user_weibo_count',
#             'folfans_ratio',
#             'user_description',
#             'image_width',
#             'image_height',
#             'image_kb',
#             'pca_color_moment1',
#             'pca_net1',
#             'pca_net3',
#             'pca_net8',
#             'label']
# features_list = []
# for i in range(1, 11):
#     features_list.append('word2vec_' + str(i))
# df_hah = pd.read_csv('G:/new_wtv.csv',names=features_list)
# print(df_hah.shape)
# df = read_data_frame(fusion_no_object_csv_path, use_cols=selected_features)[18200:20200].reset_index(drop=True)
# df = read_data_frame(fusion_no_object_csv_path, use_cols=selected_features)
# # print(df.shape)
# # df_new = pd.concat([df, df_hah],  axis=1)
# # print(df_new.shape)
# df, estimator = rf_classifier(df)
# print(df_r.shape)
# 随机森林ACC：
#  0.892
# 随机森林F 1：
#  0.8930325083565701
# 随机森林AUC：
#  0.8915698856304467

# 随机森林ACC：
#  0.9
# 随机森林F 1：
#  0.9003893097643098
# 随机森林AUC：
#  0.8936027293136057