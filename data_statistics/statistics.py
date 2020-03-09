import pandas as pd
import numpy as np
import json
import pandas_profiling


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


user_csv_path = r'G:\毕设\数据集\微博\user.csv'
def user_data_read():
    '''
    用户特征文件的读取
    :return: 用户特征文件
    '''
    df_user = pd.read_csv(user_csv_path)
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

df_user = user_data_read()
print("hah")

# 数据集分析函数
profile = df_user.profile_report(title='Titanic Dataset')
profile.to_file(output_file='G:/report2.html')
