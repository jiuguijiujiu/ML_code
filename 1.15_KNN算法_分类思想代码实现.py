# 1. 导包
from sklearn.neighbors import KNeighborsClassifier

# 2. 准备测试集（训练集 与 测试集）
x_train = [[0], [1], [2], [2]]          # 训练集特征,因为特征可以有多个，所以要写成二维的
y_train = [0, 0, 1, 1]                  # 训练集标签，标签写一维的
x_test = [[5]]                          # 测试集特征

# 3. 创建（knn分类模型）模型对象
estimator = KNeighborsClassifier(n_neighbors = 2)

# 4. 模型训练
estimator.fit(x_train, y_train)

# 5. 模型预测
y_pre = estimator.predict(x_test)

# 打印结果
print(y_pre)