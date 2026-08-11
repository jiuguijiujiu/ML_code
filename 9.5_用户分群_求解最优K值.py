"""
案例：
    基于用户的 年收入 和 消费质数，根据用户的 相似性 进行 聚类，
"""

# 导包
import os
os.environ['OMP_NUM_THREADS'] = '1'    # 设置OMP程序运行时使用的线程数

# 导包
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, silhouette_score

# 1. 加载数据
df = pd.read_csv('./data/customers.csv')
# df.info()
# print(df.head())

# 2. 定义sse与sc列表，记录不同k值时模型评估效果
sse_list = []
sc_list = []

# 3. 抽取特征
x = df.iloc[:, 3:]

# 4. 定义for循环训练，记录不同k值的模型效果
for k in range(2, 20):
    # 3.1 定义KMeans模型对象
    estimator = KMeans(n_clusters = k, max_iter = 100, random_state = 23)
    # 3.2 模型训练与预测
    y_pred = estimator.fit_predict(x)
    # 模型评估记录
    sse_list.append(estimator.inertia_)
    sc_list.append(silhouette_score(x, y_pred))

# 5. 绘制折现图，看看k值为多少时，模型最好
plt.figure(figsize=(20, 10))
plt.plot(range(2, 20), sse_list, label = 'SSE')
plt.show()
plt.plot(range(2, 20), sc_list, label = 'SC')
plt.show()