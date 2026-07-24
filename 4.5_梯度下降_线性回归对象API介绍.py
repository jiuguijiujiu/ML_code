# 导包
# from sklearn.datasets import load_boston    # 数据，因为报错，注释了
from sklearn.preprocessing import StandardScaler  # 特征处理
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.linear_model import LinearRegression   # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor    # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error  # 均方误差，均方根误差，平均绝对误差进行评估

# 1. 加载数据
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])          # hstack()函数作用：水平拼接数组
target = raw_df.values[1::2, 2]

# print(f'特征：{data.shape}')           # 特征：(506, 13)
# print(f'特征：{target.shape}')         # 特征：(506,)
# print(f'特征数据：{data[:5]}')
# print(f'标签数据：{target[:5]}')

# 2. 数据预处理：切分数据集与测试集
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=8)

# 3. 特征工程（特征提取，特征预处理...）
# 3.1 创建标准化对象
transfor = StandardScaler()
# 3.2 标准化 训练集与测试集 的特征数据
x_train = transfor.fit_transform(x_train)
x_test = transfor.transform(x_test)

# 4. 模型训练
# 4.1 创建 随机梯度下降线性回归 模型对象
# 参数1：是否需要截距(bias，偏置)，默认True
# 参数2：学习率策略，这里设置constant（不变）
# 参数3：学习率，常用0.001—0.01之间
estimator = SGDRegressor(fit_intercept=True, learning_rate='constant',eta0=0.01,)
# 4.2 模型训练
estimator.fit(x_train, y_train)
# 4.2 查看线性回归模型训练出来的 权重与偏置
print(f'权重：{estimator.coef_}')
print(f'权重：{estimator.intercept_}')

# 5. 模型预测
y_pre = estimator.predict(x_test)
print(f'预测结果为：{y_pre}')

# 6. 模型评估
# 参数1：测试集的标签
# 参数2：预测结果
print(f'均方误差：{mean_squared_error(y_test, y_pre)}')
print(f'均方根误差：{root_mean_squared_error(y_test, y_pre)}')
print(f'平均绝对误差：{mean_absolute_error(y_test, y_pre)}')



