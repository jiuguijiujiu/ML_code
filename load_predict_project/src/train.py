import sys
from pathlib import Path

# 将项目根目录 load_predict_project 加入 sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 导包
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, \
    mean_absolute_percentage_error
import joblib
import numpy as np

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15

# 1. 定义电力负荷模型类，配置日志，获取数据源。
class PowerLoadModel:
    # 1.1 初始化属性信息
    def __init__(self, file_path):
        # 1.2 拼接日志属性名
        logfile_name = 'train_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # 1.3 获取日志对象
        self.logfile = Logger('../', logfile_name).get_logger()
        # 测试写一条日志
        self.logfile.info('开始创建电力负荷模型对象')
        # 1.4 获取数据
        self.data_source = data_preprocessing(file_path)

# 2. 查看数据的整体分布情况。
def ana_data(data):         # analysis:分析
    """
    1. 查看数据整体情况
    2. 负荷整体的分布情况
    3. 各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    4. 各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    5. 工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    :param data: 数据源
    :return:
    """
    # 0.  为了防止会修改原数据，我们进行一次拷贝
    ana_data = data.copy()
    # 1. 查看数据整体情况
    ana_data.info()
    # 2. 负荷整体的分布情况,直方图
    # 2.1 创建画布
    fig = plt.figure(figsize = (20,40))
    # 2.2 添加子图
    ax1 = fig.add_subplot(411)
    ax1.hist(ana_data['power_load'], bins = 100)        # 负荷，直方图，100个区间
    ax1.set_title('负荷整体分布情况')
    ax1.set_xlabel('负荷')

    # 3. 各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    # 3.1 新增一列，充当小时
    ana_data['hour'] = ana_data['time'].str[11:13]
    # print(ana_data.head())
    # 3.2 根据小时分组，计算平均值
    hour_load_mean = ana_data.groupby(['hour'], as_index = False)['power_load'].mean()
    # print(hour_load_mean.head())      # [列1 hour, 列2 power_load 当前小时的平均负荷]
    # 3.3 画折线图
    ax2 = fig.add_subplot(412)
    ax2.plot(hour_load_mean['hour'], hour_load_mean['power_load'])
    ax2.set_title('各个小时的平均负荷趋势')
    ax2.set_xlabel('小时')
    ax2.set_ylabel('平均负荷')

    # 4. 各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    # 4.1 新增一列，充当月
    ana_data['month'] = ana_data['time'].str[5:7]
    # 4.2 根据月分组，计算平均值
    month_load_mean = ana_data.groupby(ana_data['month'], as_index = False)['power_load'].mean()
    # 4.3 print(month_load_mean.head())      # [列1 month, 列2 power_load 每月的平均负荷]
    # 画折线图
    ax3 = fig.add_subplot(413)
    ax3.plot(month_load_mean['month'], month_load_mean['power_load'])
    ax3.set_title('每个月的平均负荷趋势')
    ax3.set_xlabel('月')
    ax3.set_ylabel('平均负荷')

    # 5. 工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    # 5.1 新增一列，充当weekday
    ana_data['weekday'] = ana_data['time'].apply(lambda x:pd.to_datetime(x).weekday())
    # print(ana_data.head())
    # ana_data['is_holiday'] = np.where(ana_data['weekday'] >4, 1, 0)    # 写法1
    ana_data['is_holiday'] = ana_data['weekday'].apply(lambda x:1 if x in [5, 6] else 0)
    # print(ana_data.head(50))
    work_load_mean = ana_data[ana_data['is_holiday'] == 0].power_load.mean()
    holiday_load_mean = ana_data[ana_data['is_holiday'] == 1].power_load.mean()
    ax4 = fig.add_subplot(414)
    ax4.bar(['工作日', '周末'], [work_load_mean, holiday_load_mean])
    ax3.set_title('工作日与周末平均负荷趋势')

    # 6. 图片查看与保存
    plt.show()
    # plt.savefig('../data/fig/电力负荷数据分析图.png')

# 3. 特征工程(重点)
def feature_engineering(data, logger):
    """
    对给定的数据源，进行特征工程处理，提取出关键的特征
    1. 提取出时间特征：小时、月份
    2. 提取出相近时间窗口中的负荷特征：step 大小窗口的负荷
    3. 提取昨日同时刻负荷特征
    4. 删除出现空值的样本
    5. 整理时间特征，并返回
    :param data: 数据源
    :param logger: 日志
    :return:
    """
    # 为了防止会修改原数据，我们进行一次拷贝
    feature_data = data.copy()
    # 1. 提取出时间特征：小时、月份
    # 提取小时
    feature_data['hour'] = feature_data['time'].str[11:13]
    # 提取月份
    feature_data['month'] = feature_data['time'].str[5:7]
    # 热编码one-hot处理hour,mouth字段
    # hour_month_data = pd.get_dummies(feature_data, columns=['hour', 'month'])        #写法1
    hour_month_data = pd.get_dummies(feature_data[['hour', 'month']])
    # print(hour_month_data.head(10))
    # hour_month_data.info()
    # 拼接
    feature_data = pd.concat([feature_data, hour_month_data], axis = 1)
    # print(feature_data.head())
    # feature_data.info()

    # 2. 提取出相近时间窗口中的负荷特征：step 大小窗口的负荷
    # 2.1 shift从头向下空x格，那同一列的数据就是前x个小时的电力负荷
    # 注意这是前x小时的负荷，不是前x小时负荷之和
    load_1h_data = feature_data['power_load'].shift(1)  # 前1h的负荷
    load_2h_data = feature_data['power_load'].shift(2)  # 前2h的负荷
    load_3h_data = feature_data['power_load'].shift(3)  # 前3h的负荷
    # 2.2 拼接
    load_shift_data = pd.concat([load_1h_data, load_2h_data, load_3h_data], axis = 1)
    # 2.3 修改列名
    load_shift_data.columns = ['前1小时', '前2小时', '前3小时']
    # 2.4 拼接
    feature_data = pd.concat([feature_data, load_shift_data], axis = 1)
    # feature_data.info()

    # 3. 提取昨日同时刻负荷特征
    # 3.1 新增一列，昨天的时间yesterday_time
    #strftime('%Y-%m-%d %H:%M:%S')将datetime类型转为str类型，否则下面字典找不到，因为类型不匹配
    feature_data['yesterday_time'] = feature_data['time'].apply(lambda x:(pd.to_datetime(x) - datetime.timedelta(days = 1)).strftime('%Y-%m-%d %H:%M:%S'))
    # print(feature_data.head())
    # 3.2 我们把所有的 日期 和 负荷 拼接成字典，方便查找。
    time_load_dict = feature_data.set_index('time')['power_load'].to_dict()
    # print(time_load_dict)
    # 格式：{'2013-09-02 00:00:00': 750.75, '2013-09-02 01:00:00': 716.94, '2013-09-02 02:00:00': 712.77, ...}
    # 3.3 新增1列 yesterday_load，表示：昨天的相同时刻的负荷。
    feature_data['yesterday_load'] = feature_data['yesterday_time'].apply(lambda x:time_load_dict.get(x))
    # print(feature_data.head(30))
    # feature_data.info()

    # 4. 删除出现空值的样本
    feature_data = feature_data.dropna()
    # 5. 整理时间特征，并返回
    feature_columns = list(hour_month_data.columns) + list(load_shift_data.columns) + ['yesterday_load']
    # print(feature_columns)
    return feature_data, feature_columns

# 4. 模型训练，评估，保存
def model_train(data, features, logger):
    """
    1.数据集切分
    2.网格化搜索与交叉验证
    3.模型实例化
    4.模型训练
    5.模型评价
    6.模型保存
    :param data: 特征工程处理后的输入数据
    :param features: 特征名称
    :param logger: 日志对象
    :return:
    """
    # 1.数据集切分
    x = data[features]
    y = data['power_load']
    # print(x.shape, y.shape)
    # print(x.head())
    # print(y.head())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 23)

    # # 2.网格化搜索与交叉验证
    # logger.info("-----网格搜索 + 交叉验证 寻找最优超参-----")
    # logger.info(f'开始时间：{datetime.datetime.now()}')
    # # 2.1 定义参数字典
    # param_dict = {
    #     'n_estimators': [50, 100, 150, 200],
    #     'max_depth': [3, 5, 6, 7],
    #     'learning_rate': [0.01, 0.1]
    # }
    # # 2.2 创建XGBRegressor模型对象(Extreme Gradient Boosting Tree, 极限梯度提升树)
    # estimator = XGBRegressor(random_state=23)
    # # 2.3 创建网格搜索模型对象
    # gs = GridSearchCV(estimator, param_grid=param_dict, cv = 5)
    # # 2.4 模型训练
    # gs.fit(x_train, y_train)
    # # 2.5 打印最优参数组合
    # logger.info(f'最优参数组合：{gs.best_params_}')
    # logger.info(f'结束时间：{datetime.datetime.now()}')

    # 3.模型实例化
    estimator = XGBRegressor(n_estimators = 200, max_depth = 5, learning_rate = 0.1)
    # 4.模型训练
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    # 5.模型评价
    print(f'均方误差：{mean_squared_error(y_test, y_pred)}')
    print(f'平均绝对误差：{mean_absolute_error(y_test, y_pred)}')
    print(f'均方根误差：{root_mean_squared_error(y_test, y_pred)}')
    print(f'平均绝对百分比误差:{mean_absolute_percentage_error(y_test, y_pred)}')


    # 6.模型保存
    joblib.dump(estimator, '../model/power_load_model.pkl')       # pickle文件 -> 后缀名一般是.pkl, .pth .pickle
    logger.info('-----模型保存成功-----')

# 5. 测试
if __name__ == '__main__':
    # 5.1 创建电力负荷模型对象
    pm = PowerLoadModel('../data/train.csv')
    # 5.2 打印数据源
    # print(pm.data_source)
    # 5.3 调用查看数据分布
    # ana_data(pm.data_source)
    # 5.4 特征工程
    feature_data, feature_columns = feature_engineering(pm.data_source, pm.logfile)
    # 5.5 模型训练
    # 参1：处理后的全部数据集， 参2：特征名称列表， 参3：日志对象。
    model_train(feature_data, feature_columns, pm.logfile)