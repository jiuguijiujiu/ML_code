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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import numpy as np

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15

# 1. 定义电力负荷模型类，配置日志，获取数据源。
class PowerLoadModel:
    # 1.1 初始化属性信息
    def __init__(self):
        # 1.2 拼接日志属性名
        logfile_name = 'train_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # 1.3 获取日志对象
        self.logfile = Logger('../', logfile_name).get_logger()
        # 测试写一条日志
        self.logfile.info('开始创建电力负荷模型对象')
        # 1.4 获取数据
        self.data_source = data_preprocessing()

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
    # ana_data['is_holiday'] = np.where(ana_data['weekday'] >4, True, False)    # 写法1
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
    # 3. 提取昨日同时刻负荷特征
    # 4. 删除出现空值的样本
    # 5. 整理时间特征，并返回


# 4. 模型训练，评估。


# 5. 测试
if __name__ == '__main__':
    # 4.1 创建电力负荷模型对象
    pm = PowerLoadModel()
    # 4.2 打印数据源
    # print(pm.data_source)
    # 4.3 调用查看数据分布
    # ana_data(pm.data_source)
    # 4.4 特征工程
    feature_engineering(pm.data_source, pm.logfile)