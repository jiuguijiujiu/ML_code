# 1. 导包
from sklearn.neighbors import KNeighborsClassifier

# 2. 准备数据
x_train = [[0], [1], [2], [3], [4], [5]]
y_train = [0, 1, 0, 1, 1, 1]
x_test = [[4]]

# 3. 创建模型对象
estimator = KNeighborsClassifier(n_neighbors=3)

# 4. 模型训练
estimator.fit(x_train, y_train)

# 5. 模型预测
y_pre = estimator.predict(x_test)

# 输出结果
print(y_pre)