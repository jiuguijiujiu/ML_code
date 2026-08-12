import os
os.environ['OMP_NUM_THREADS'] = '1'    # 设置OMP程序运行时使用的线程数

# 导包
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载数据集
df = pd.read_csv('./data/customers.csv')

# 2. 提取特征
x = df.iloc[:, 3:5]
# print(x.head())
# print(x.values)                       # [[15, 39], [15, 81], [16, 6], [16, 77], [17, 40], [17, 76]]

# 3. 创建模型对象
estimator = KMeans(n_clusters = 5, max_iter = 100, random_state = 23)

# 4. 模型训练与预测
y_pred = estimator.fit_predict(x)
# print(y_pred)                         # [2, 3, 2, 3, 2, 0, 0, 1, 4, 4, 0]

# 5. 绘制 5个簇的 样本点 → 散点图
# 用x训练完，x里有y_pred列?
plt.scatter(x.values[y_pred == 0, 0], x.values[y_pred == 0, 1], label = 'Cluster 1')        # 0号簇
plt.scatter(x.values[y_pred == 1, 0], x.values[y_pred == 1, 1], label = 'Cluster 2')        # 1号簇
plt.scatter(x.values[y_pred == 2, 0], x.values[y_pred == 2, 1], label = 'Cluster 3')        # 2号簇
plt.scatter(x.values[y_pred == 3, 0], x.values[y_pred == 3, 1], label = 'Cluster 4')        # 3号簇
plt.scatter(x.values[y_pred == 4, 0], x.values[y_pred == 4, 1], label = 'Cluster 5')        # 4号簇

# 6. 绘制 5个簇的 质心 → 散点图
# print(estimator.cluster_centers_)           # 5个质心的坐标.
plt.scatter(estimator.cluster_centers_[:, 0], estimator.cluster_centers_[:, 1])

# 设置标题,标签,图例
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()



# 代码 x.values[y_pred == 0, 0] 解释.
x = [[15, 39], [15, 81], [16, 6], [16, 77], [17, 40], [17, 76]]
x2 = np.array(x)        # 模拟: x.values, 即: <class 'numpy.ndarray'> 对象
# print(x2)

# 模拟: x.values[y_pred == 0]
# 细节: 随便写, 个数要一致, 即: True是要, False是不要.
result = x2[[True, False, True, True, False, False]]      # [[15, 39], [16, 6], [16, 77]]
# print(result)

# 模拟: x.values[y_pred == 0, 0]
result2 = x2[[True, False, True, True, False, False], 0]      # [[15, 39], [16, 6], [16, 77]]
print(result2)

# 模拟: x.values[y_pred == 0, 1]
result3 = x2[[True, False, True, True, False, False], 1]      # [[15, 39], [16, 6], [16, 77]]
print(result3)