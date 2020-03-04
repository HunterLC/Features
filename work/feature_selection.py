from sklearn.feature_selection import RFECV
import pandas as pd
import numpy as np
from sklearn import metrics, model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

fusion_csv_path = r'G:\毕设\数据集\微博\fusion_news_features.csv'
text_csv_path = r'G:\毕设\数据集\微博\text.csv'
user_csv_path = r'G:\毕设\数据集\微博\user.csv'
image_csv_path = r'G:\毕设\数据集\微博\image.csv'

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
    #预设各参数的不同选项值
    max_depth = [18, 19, 20, 21, 22] #数据量小在10左右，数据量大在20左右
    min_samples_split = [2, 4, 6, 8]
    min_samples_leaf = [2, 4, 8, 10, 12]
    #将各参数值以字典形式组织起来
    parameters = {'max_depth': max_depth, 'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf}
    # 网格搜索法,测试不同的参数值
    grid_dtcateg= GridSearchCV(estimator = RandomForestClassifier(),
                               param_grid = parameters, cv = 10)
    #模型拟合
    grid_dtcateg.fit(X_train, y_train)
    #返回最佳组合的参数值
    print(grid_dtcateg.best_params_)
    return grid_dtcateg.best_params_

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
    estimator = RandomForestClassifier(max_depth=20,
                                       min_samples_leaf=4,
                                       min_samples_split=6,
                                       n_estimators=100,
                                       bootstrap=True,
                                       max_features='sqrt',
                                       verbose=1,
                                       n_jobs=-1)

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
    # rfe_model_rf = RFECV(estimator, step=1000, cv=10, scoring=None, verbose=1, n_jobs=-1)
    # rfe_model_rf = rfe_model_rf.fit(X_train, y_train)
    # rf_pred = rfe_model_rf.predict(X_test)
    # print('随机森林准确率：\n', metrics.accuracy_score(y_test, rf_pred))
    # return rfe_model_rf
    estimator = estimator.fit(X_train, y_train)
    rf_pred = estimator.predict(X_test)
    print('随机森林ACC：\n', metrics.accuracy_score(y_test, rf_pred))
    print('随机森林F 1：\n', metrics.f1_score(y_test, rf_pred, average='weighted'))
    print('随机森林AUC：\n', metrics.roc_auc_score(y_test, rf_pred))
    # 绘制ROC曲线，一般认为AUC大于0.8即算较好效果
    draw_auc(y_test, rf_pred)
    # 绘制混淆矩阵热力图
    draw_confusion_matrix_heat_map(y_test, rf_pred)
    return estimator

def draw_auc(y_test, rf_pred):
    # 计算绘图数据
    fpr, tpr, threshold = metrics.roc_curve(y_test, rf_pred)
    roc_auc = metrics.auc(fpr, tpr)
    # 绘图
    plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
    plt.plot(fpr, tpr, color='black', lw=1)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.text(0.5, 0.3, 'ROC Curve (area = %0.2f)' % roc_auc)
    plt.xlabel('specificity')
    plt.ylabel('sensitivity')
    plt.show()

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

model = rf_classifier(fusion_csv_path)

# i = 0
# for item in dd:
#     if model.support_.tolist()[i] == True:
#         print(item)
#     i += 1