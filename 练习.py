# 导包
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 读取数据
data = pd.read_csv('./data/breast-cancer-wisconsin.csv')
# data.info()
# print(data.describe())
# print(data.head())

# 数据预处理
data.replace('?', np.nan, inplace = True)
data.dropna(inplace = True)
# data.info()

# 特征工程（提取，预处理，选择，降维，组合）
# 提取特征与标签
x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 888)

# 正则化
transfor = StandardScaler()
x_train = transfor.fit_transform(x_train)
x_test = transfor.transform(x_test)
# print(x_train[:5])

# 模型训练
estimator = LogisticRegression()
estimator.fit(x_train, y_train)
y_pre = estimator.predict(x_test)
print(y_pre)

# 模型预测
print(estimator.score(x_test, y_test))
print(accuracy_score(y_test, y_pre))