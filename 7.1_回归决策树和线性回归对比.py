# 导包
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor # 回归决策树
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. 准备数据
# 训练集特征
x_train = np.array(list(range(1, 11))).reshape(-1, 1)
# 训练集标签
y_train = np.array([5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9, 9.05])
# print(x_train)
# print(y_train)

# 2. 数据预处理

# 3. 特征工程

# 4. 模型训练
# 4.1 创建模型对象
estimator1 = LinearRegression()                         # 线性回归
estimator2 = DecisionTreeRegressor(max_depth = 1)       # 回归决策树，最大树深为1
estimator3 = DecisionTreeRegressor(max_depth = 3)       # 回归决策树，最大树深为3
# 4.2 训练
estimator1.fit(x_train, y_train)
estimator2.fit(x_train, y_train)
estimator3.fit(x_train, y_train)

# 5. 模型预测
# 5.1 准备测试集的特征数据
# x_test = np.array(list(range(0, 10, 0.1))).reshape(-1, 1)       # 报错，python自带的range（）并支持小数
x_test = np.arange(0, 10, 0.1).reshape(-1, 1)
# print(x_test)
# 5.2 预测
y_pred1 = estimator1.predict(x_test)
y_pred2 = estimator2.predict(x_test)
y_pred3 = estimator3.predict(x_test)
# 打印
# print(f'线性回归预测结果：{y_pred1}')
# print(f'深度为1，回归决策树预测结果：{y_pred2}')
# print(f'深度为3，回归决策树预测结果：{y_pred3}')

# 6. 模型评估

# 7. 绘图
# 7.1 以真实值(训练集)绘制散点图
plt.scatter(x = x_train, y = y_train, color = 'gray')
# 7.2 以预测值（线性回归，回归决策树）绘制折线图
plt.plot(x_test, y_pred1, color = 'red', label = 'linear_regression')
plt.plot(x_test, y_pred2, color = 'blue', label = 'max_deep = 1')
plt.plot(x_test, y_pred3, color = 'green', label = 'max_deep = 3')
# 7.3 显示图例
plt.legend()
# 7.4 设置标题，x,y轴标签
plt.title('Decision Tree Regressor')
plt.xlabel('data')
plt.ylabel('target')

plt.show()