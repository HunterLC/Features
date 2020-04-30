import pandas as pd
import numpy as np
import json
import pandas_profiling
import matplotlib.pyplot as plt


# #数据集读入
# news_data = pd.read_csv(r'G:\毕设\数据集\微博\train.csv')
# #查看数据集是否存在缺失值
# # id                     0
# # text                   0
# # piclist            16639
# # userGender           445
# # userFollowCount      464
# # userFansCount        448
# # userWeiboCount       472
# # userLocation         537
# # userDescription     7715
# # category               0
# # label                  0
# news_data.apply(lambda x:np.sum(x.isnull()))
# #查看连续型数据的描述性统计值  userFollowCount  userFansCount  userWeiboCount label  4
# news_data.describe()
# #查看离散型数据的描述性统计值
# news_data.describe(include=['object'])
# #查看所有数据 38471 rows x 11 columns
# news_data.all

fusion_no_object_csv_path = r'G:\毕设\数据集\微博\fusion_features_0306_no_object.csv'
user_csv_path = r'G:\毕设\数据集\微博\train.csv'
def user_data_read():
    '''
    用户特征文件的读取
    :return: 用户特征文件
    '''
    df_user = pd.read_csv(fusion_no_object_csv_path)
    return df_user

def get_user_location(df_user):
    unknown = 0 #未知
    others = 0  #其他
    overseas = 0 #海外
    BJ = 0  #北京
    TJ = 0  #天津
    SH = 0  #上海
    CQ = 0  #重庆
    HE = 0  #河北
    SX = 0  #山西
    SD = 0  #山东
    NM = 0  #内蒙古
    LN = 0  #辽宁
    JL = 0  #吉林
    HL = 0  #黑龙江
    JS = 0  #江苏
    ZJ = 0  #浙江
    AH = 0  #安徽
    FJ = 0  #福建
    JX = 0  #江西
    HA = 0  #河南
    HB = 0  #湖北
    HN = 0  #湖南
    GD = 0  #广东
    GX = 0  #广西
    HI = 0  #海南
    SC = 0  #四川
    GZ = 0  #贵州
    YN = 0  #云南
    XZ = 0  #西藏
    SN = 0  #陕西
    GS = 0  #甘肃
    QH = 0  #青海
    NX = 0  #宁夏
    XJ = 0  #新疆
    HongKong= 0  #香港
    Macao= 0  #澳门
    Taiwan= 0  #台湾
    for index, row in df_user.iterrows():
        user_location = row['user_location']
        if(pd.isna(user_location)):
            unknown += 1
        elif(user_location[0:2] == '其他'):
            others += 1
        elif (user_location[0:2] == '海外'):
            overseas += 1
        elif (user_location[0:2] == '北京'):
            BJ += 1
        elif (user_location[0:2] == '天津'):
            TJ += 1
        elif (user_location[0:2] == '上海'):
            SH += 1
        elif (user_location[0:2] == '重庆'):
            CQ += 1
        elif (user_location[0:2] == '河北'):
            HE += 1
        elif (user_location[0:2] == '山西'):
            SX += 1
        elif (user_location[0:2] == '山东'):
            SD += 1
        elif (user_location[0:2] == '内蒙'):
            NM += 1
        elif (user_location[0:2] == '辽宁'):
            LN += 1
        elif (user_location[0:2] == '吉林'):
            JL += 1
        elif (user_location[0:2] == '黑龙'):
            HL += 1
        elif (user_location[0:2] == '江苏'):
            JS += 1
        elif (user_location[0:2] == '浙江'):
            ZJ += 1
        elif (user_location[0:2] == '安徽'):
            AH += 1
        elif (user_location[0:2] == '福建'):
            FJ += 1
        elif (user_location[0:2] == '江西'):
            JX += 1
        elif (user_location[0:2] == '河南'):
            HA += 1
        elif (user_location[0:2] == '湖北'):
            HB += 1
        elif (user_location[0:2] == '湖南'):
            HN += 1
        elif (user_location[0:2] == '广东'):
            GD += 1
        elif (user_location[0:2] == '广西'):
            GX += 1
        elif (user_location[0:2] == '海南'):
            HI += 1
        elif (user_location[0:2] == '四川'):
            SC += 1
        elif (user_location[0:2] == '贵州'):
            GZ += 1
        elif (user_location[0:2] == '云南'):
            YN += 1
        elif (user_location[0:2] == '西藏'):
            XZ += 1
        elif (user_location[0:2] == '陕西'):
            SN += 1
        elif (user_location[0:2] == '甘肃'):
            GS += 1
        elif (user_location[0:2] == '青海'):
            QH += 1
        elif (user_location[0:2] == '宁夏'):
            NX += 1
        elif (user_location[0:2] == '新疆'):
            XJ += 1
        elif (user_location[0:2] == '香港'):
            HongKong += 1
        elif (user_location[0:2] == '澳门'):
            Macao += 1
        elif (user_location[0:2] == '台湾'):
            Taiwan += 1
    map = [{"name":"北京","value":BJ},
           {"name":"天津","value":TJ},
           {"name":"上海","value":SH},
           {"name":"重庆","value":CQ},
           {"name":"河北","value":HE},
           {"name":"山西","value":SX},
           {"name": "山东", "value": SD},
           {"name":"内蒙古","value":NM},
           {"name":"辽宁","value":LN},
           {"name":"吉林","value":JL},
           {"name":"黑龙江","value":HL},
           {"name":"江苏","value":JS},
           {"name":"浙江","value":ZJ},
           {"name":"安徽","value":AH},
           {"name":"福建","value":FJ},
           {"name":"江西","value":JX},
           {"name":"河南","value":HA},
           {"name":"湖北","value":HB},
           {"name":"湖南","value":HN},
           {"name":"广东","value":GD},
           {"name":"广西","value":GX},
           {"name":"海南","value":HI},
           {"name":"四川","value":SC},
           {"name":"贵州","value":GZ},
           {"name":"云南","value":YN},
           {"name":"西藏","value":XZ},
           {"name":"陕西","value":SN},
           {"name":"甘肃","value":GS},
           {"name":"青海","value":QH},
           {"name":"宁夏","value":NX},
           {"name":"新疆","value":XJ},
           {"name":"香港","value":HongKong},
           {"name":"澳门","value":Macao},
           {"name":"台湾","value":Taiwan},
           {"name": "南海诸岛", "value": 0},
           {"name":"未知","value":unknown},
           {"name":"海外","value":overseas},
           {"name":"其他","value":others}]
    return map

def get_user_gender(df_user):
    """
    df_user['user_gender'].value_counts()一句就搞定了，我真沙雕
    :param df_user:
    :return:
    """
    unknown = 0 #未知
    male = 0 #男性
    female = 0 #女性
    for index, row in df_user.iterrows():
        user_gender = row['user_gender']
        if (pd.isna(user_gender)):
            unknown += 1
        elif (user_gender == '男'):
            male += 1
        elif (user_gender == '女'):
            female += 1
    list_gender = [male, female, unknown]
    list_gender = map(lambda x: str(x), list_gender)
    response = {"status": 200,
                "gender": ','.join(list_gender)}
    return response

# df_user = user_data_read()
# # 数据集分析函数
# profile = df_user.profile_report(title='虚假新闻检测数据集')
# profile.to_file(output_file='G:/111.html')

def draw_gender_true_fake(df):
    m_t = 0
    m_f = 0
    w_t = 0
    w_f = 0
    u_t = 0
    u_f = 0
    for index, row in df.iterrows():
        if row['userGender'] == '男':
            if row['label'] == 0:
                m_t += 1
            else:
                m_f += 1
        elif row['userGender'] == '女':
            if row['label'] == 0:
                w_t += 1
            else:
                w_f += 1
        elif pd.isna(row['userGender']):
            if row['label'] == 0:
                u_t += 1
            else:
                u_f += 1
    print(str(m_t))
    print(str(m_f))
    print(str(w_t))
    print(str(w_f))
    print(str(u_t))
    print(str(u_f))
    label_list = ['男', '女', '未知']
    num_list1 = [13484, 5378, 324]
    num_list2 = [9123, 10041, 121]
    x = range(len(num_list1))
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """
    rects1 = plt.bar(x=x, height=num_list1, width=0.4, alpha=0.8, color='steelblue', label="真实新闻")
    rects2 = plt.bar(x=[i + 0.4 for i in x], height=num_list2, width=0.4, color='indianred', label="虚假新闻")
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + 0.2 for index in x], label_list)
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.legend(fontsize=20)
    # 设置刻度字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('性别', fontsize=15)
    plt.ylabel('样本数', fontsize=15)
    plt.title(u'')
    plt.show()

def draw_fol_cdf(df):
    df.fillna(0, inplace=True)
    t_list = []
    f_list = []
    temp1 = []
    temp2 = []
    for index, row in df.iterrows():
        if row['label'] == 0: # 真新闻
            temp1.append(float(row['userFollowCount']))
        else:
            temp2.append(float(row['userFollowCount']))
    temp1.sort()
    temp2.sort()
    count1 = len(temp1)
    count2 = len(temp2)
    for i in range(count1):
        t_list.append((i+1)/count1)
    for j in range(count2):
        f_list.append((j + 1) / count2)
        # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.plot(temp1, t_list, linewidth=2, alpha=0.8, linestyle='-', color='steelblue', label="真实新闻")
    plt.plot(temp2, f_list, linewidth=2, alpha=0.8, linestyle='-', color='indianred', label="虚假新闻")
    plt.legend(fontsize=20)
    # 设置刻度字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('关注数', fontsize=15)
    plt.ylabel('CDF值', fontsize=15)
    plt.show()

def draw_fans_cdf(df):
    df.fillna(0, inplace=True)
    t_list = []
    f_list = []
    temp1 = []
    temp2 = []
    for index, row in df.iterrows():
        if row['label'] == 0: # 真新闻
            temp1.append(float(row['userFansCount']))
        else:
            temp2.append(float(row['userFansCount']))
    temp1.sort()
    temp2.sort()
    count1 = len(temp1)
    count2 = len(temp2)
    for i in range(count1):
        t_list.append((i+1)/count1)
    for j in range(count2):
        f_list.append((j + 1) / count2)
        # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.plot(temp1, t_list, linewidth=2, alpha=0.8, linestyle='-', color='steelblue', label="真实新闻")
    plt.plot(temp2, f_list, linewidth=2, alpha=0.8, linestyle='-', color='indianred', label="虚假新闻")
    plt.legend(fontsize=20)
    # 设置刻度字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('粉丝数', fontsize=15)
    plt.ylabel('CDF值', fontsize=15)
    plt.show()

def draw_weibo_cdf(df):
    df.fillna(0, inplace=True)
    t_list = []
    f_list = []
    temp1 = []
    temp2 = []
    for index, row in df.iterrows():
        if row['label'] == 0: # 真新闻
            temp1.append(float(row['userWeiboCount']))
        else:
            temp2.append(float(row['userWeiboCount']))
    temp1.sort()
    temp2.sort()
    count1 = len(temp1)
    count2 = len(temp2)
    for i in range(count1):
        t_list.append((i+1)/count1)
    for j in range(count2):
        f_list.append((j + 1) / count2)
        # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.plot(temp1, t_list, linewidth=2, alpha=0.8, linestyle='-', color='steelblue', label="真实新闻")
    plt.plot(temp2, f_list, linewidth=2, alpha=0.8, linestyle='-', color='indianred', label="虚假新闻")
    plt.legend(fontsize=20)
    # 设置刻度字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('发博数', fontsize=15)
    plt.ylabel('CDF值', fontsize=15)
    plt.show()

def draw_4_3():
    label_list = ['FO', 'WPR', 'EXB','X2-RFE-RF']
    num_list1 = [75.662, 436.961, 688.478, 343.727]
    x = range(len(num_list1))
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """
    rects1 = plt.bar(x=x, height=num_list1, width=0.4, alpha=0.8, color='steelblue')
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + 0 for index in x], label_list)
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.legend(fontsize=20)
    # 设置刻度字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('特征选择算法类别', fontsize=15)
    plt.ylabel('算法运行时间', fontsize=15)
    plt.title(u'')
    plt.show()

def draw_4_4_1():
    label_list = ['决策树', 'KNN', '随机森林']
    num_list1 = [88.452, 77.372, 93.245]
    num_list2 = [89.223, 78.161, 94.754]
    num_list3 = [89.452, 78.342, 94.564]
    num_list4 = [89.403, 78.541, 94.353]
    x = range(len(num_list1))
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """
    rects1 = plt.bar(x=x, height=num_list1, width=0.2, alpha=0.8, color='steelblue', label="FO")
    rects2 = plt.bar(x=[i + 0.2 for i in x], height=num_list2, width=0.2, color='indianred', label="WPR")
    rects3 = plt.bar(x=[i + 0.4 for i in x], height=num_list3, width=0.2, color='green', label="EXB")
    rects4 = plt.bar(x=[i + 0.6 for i in x], height=num_list4, width=0.2, color='grey', label="X2-RFE-RF")
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + 0.3 for index in x], label_list)
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects3:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects4:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.legend(fontsize=10)
    # 设置刻度字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('分类模型', fontsize=15)
    plt.ylabel('ACC百分比', fontsize=15)
    plt.title(u'')
    axes = plt.gca()
    axes.set_ylim([60, 100])
    plt.show()


def draw_4_4_2():
    label_list = ['决策树', 'KNN', '随机森林']
    num_list1 = [88.451, 77.369, 93.243]
    num_list2 = [89.223, 78.144, 94.752]
    num_list3 = [89.453, 78.343, 94.560]
    num_list4 = [89.403, 78.512, 94.351]
    x = range(len(num_list1))
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """
    rects1 = plt.bar(x=x, height=num_list1, width=0.2, alpha=0.8, color='steelblue', label="FO")
    rects2 = plt.bar(x=[i + 0.2 for i in x], height=num_list2, width=0.2, color='indianred', label="WPR")
    rects3 = plt.bar(x=[i + 0.4 for i in x], height=num_list3, width=0.2, color='green', label="EXB")
    rects4 = plt.bar(x=[i + 0.6 for i in x], height=num_list4, width=0.2, color='grey', label="X2-RFE-RF")
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + 0.3 for index in x], label_list)
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects3:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects4:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.legend(fontsize=10)
    # 设置刻度字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('分类模型', fontsize=15)
    plt.ylabel('F1值百分比', fontsize=15)
    axes = plt.gca()
    axes.set_ylim([60,100])
    plt.title(u'')
    plt.show()


def draw_4_4_3():
    label_list = ['决策树', 'KNN', '随机森林']
    num_list1 = [88.453, 77.370, 93.252]
    num_list2 = [89.223, 78.147, 94.767]
    num_list3 = [89.457, 78.345, 94.579]
    num_list4 = [89.404, 78.522, 94.365]
    x = range(len(num_list1))
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """
    rects1 = plt.bar(x=x, height=num_list1, width=0.2, alpha=0.8, color='steelblue', label="FO")
    rects2 = plt.bar(x=[i + 0.2 for i in x], height=num_list2, width=0.2, color='indianred', label="WPR")
    rects3 = plt.bar(x=[i + 0.4 for i in x], height=num_list3, width=0.2, color='green', label="EXB")
    rects4 = plt.bar(x=[i + 0.6 for i in x], height=num_list4, width=0.2, color='grey', label="X2-RFE-RF")
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + 0.3 for index in x], label_list)
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects3:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects4:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.legend(fontsize=10)
    # 设置刻度字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('分类模型', fontsize=15)
    plt.ylabel('AUC百分比', fontsize=15)
    axes = plt.gca()
    axes.set_ylim([60,100])
    plt.title(u'')
    plt.show()

def draw_4_7():
    label_list = ['完整特征集', 'X2-RFE-RF选择的特征子集']
    num_list1 = [158.670, 8.412]
    x = range(len(num_list1))
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """
    rects1 = plt.bar(x=x, height=num_list1, width=0.4, alpha=0.8, color='steelblue')
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + 0 for index in x], label_list)
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.legend(fontsize=20)
    # 设置刻度字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('特征集类别', fontsize=15)
    plt.ylabel('分类运行时间', fontsize=15)
    plt.title(u'')
    plt.show()

def draw_4_9():
    label_list = ['完整特征集', 'X2-RFE-RF选择的特征子集']
    num_list1 = [90.395, 94.300]
    num_list2 = [90.373, 94.298]
    num_list3 = [90.421, 94.311]
    x = range(len(num_list1))
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """
    rects1 = plt.bar(x=x, height=num_list1, width=0.2, alpha=0.8, color='steelblue', label="ACC")
    rects2 = plt.bar(x=[i + 0.2 for i in x], height=num_list2, width=0.2, color='indianred', label="F1")
    rects3 = plt.bar(x=[i + 0.4 for i in x], height=num_list3, width=0.2, color='green', label="AUC")
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + 0.2 for index in x], label_list)
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects3:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.legend(fontsize=10)
    # 设置刻度字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('特征集类别', fontsize=15)
    plt.ylabel('随机森林指标百分比', fontsize=15)
    plt.title(u'')
    axes = plt.gca()
    axes.set_ylim([90, 100])
    plt.show()
# draw_gender_true_fake(pd.read_csv(user_csv_path,usecols=['userGender','label']))
# draw_fol_cdf(pd.read_csv(user_csv_path,usecols=['userFollowCount','label']))
# draw_fans_cdf(pd.read_csv(user_csv_path,usecols=['userFansCount','label']))
# draw_weibo_cdf(pd.read_csv(user_csv_path,usecols=['userWeiboCount','label']))
draw_4_9()