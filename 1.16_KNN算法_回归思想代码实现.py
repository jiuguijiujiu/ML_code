# 1. 导包
from sklearn.neighbors import KNeighborsRegressor

# 2. 准备数据
x_train = [[0, 0, 1],
           [1, 1, 0],
           [3, 10, 10],
           [4, 11, 12]]

y_train = [0.1, 0.2, 0.3, 0.4]
x_test = [[3, 11, 10]]

# 3. 创建knn回归算法对象
estimator = KNeighborsRegressor(n_neighbors=3)

# 4. 模型训练
estimator.fit(x_train, y_train)

# 5. 模型预测
y_pre = estimator.predict(x_test)

# 6. 输出结果
print(y_pre)