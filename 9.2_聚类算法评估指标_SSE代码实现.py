# 导包
import os
os.environ['OMP_NUM_THREADS'] = '4'
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score, silhouette_score


# 1. 定义函数，演示：SSE+肘部法
def dm01_sse():
    # 1. 定于SSE列表，记录每个K值的SSE值
    sse_list = []

    # 2. 生成数据集， 参1：样本数量， 参2：特征数量， 参3：4个簇， 参4：4个簇的std标准差， 参5：固定随机种子
    x, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
        cluster_std=[0.4, 0.2, 0.2, 0.2],
        random_state=23
    )

    # 3. for循环遍历训练，计算每个K值的SSE，并添加到每个列表中
    for k in range(1, 100):
        # 3.1 创建KMeans对象，参1：簇个数（k值），参2：最大迭代次数，参3：随机种子
        estimator = KMeans(n_clusters = k, max_iter = 100,random_state = 23)
        # 3.2 训练
        y_pred = estimator.fit(x)
        # 3.3 预测（这里不需要）
        # 3.4 计算SSE值（评估）,并添加到列表中,SSE越小越好
        sse_list.append(estimator.inertia_)

    # 4. 绘制sse曲线
    # 4.1 创建画布，指定画布尺寸
    plt.figure(figsize = (20,10))
    # 4.2 设置标题
    plt.title('SSE')
    # 4.3 设置x刻度
    plt.xticks(range(0, 100, 3))
    # 4.4 添加网格
    plt.grid()
    # 4.5 添加x轴，y轴标签
    plt.xlabel('K')
    plt.ylabel('SSE')
    # 4.6绘制折现图
    # 参1：x轴（k值），参2：y轴（SSE）
    plt.plot(range(1, 100), sse_list)
    plt.show()

# 2. 定义函数，演示：SC轮廓系数法
def dm02_sc():
    # 1. 定于sc列表，记录每个K值的sc值
    sc_list = []

    # 2. 生成数据集， 参1：样本数量， 参2：特征数量， 参3：4个簇， 参4：4个簇的std标准差， 参5：固定随机种子
    x, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
        cluster_std=[0.4, 0.2, 0.2, 0.2],
        random_state=23
    )

    # 3. for循环遍历训练，计算每个K值的sc，并添加到每个列表中
    for k in range(2, 100):     # 考虑簇外，至少2个簇
        # 3.1 创建KMeans对象，参1：簇个数（k值），参2：最大迭代次数，参3：随机种子
        estimator = KMeans(n_clusters = k, max_iter = 100,random_state = 23)
        # 3.2 训练
        y_pred = estimator.fit(x)
        # 3.3 预测
        y_pred = estimator.predict(x)
        # 3.4 计算sc值（评估）,并添加到列表中,sc越小越好
        sc_list.append(silhouette_score(x, y_pred))

    # 4. 绘制sc曲线
    # 4.1 创建画布，指定画布尺寸
    plt.figure(figsize = (20,10))
    # 4.2 设置标题
    plt.title('sc')
    # 4.3 设置x刻度
    plt.xticks(range(0, 100, 3))
    # 4.4 添加网格
    plt.grid()
    # 4.5 添加x轴，y轴标签
    plt.xlabel('K')
    plt.ylabel('sc')
    # 4.6绘制折现图
    # 参1：x轴（k值），参2：y轴（sc）
    plt.plot(range(2, 100), sc_list)
    plt.show()

# 3. 定义函数，演示：CH轮廓系数法
def dm03_ch():
    # 1. 定于ch列表，记录每个K值的ch值
    ch_list = []

    # 2. 生成数据集， 参1：样本数量， 参2：特征数量， 参3：4个簇， 参4：4个簇的std标准差， 参5：固定随机种子
    x, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
        cluster_std=[0.4, 0.2, 0.2, 0.2],
        random_state=23
    )

    # 3. for循环遍历训练，计算每个K值的ch，并添加到每个列表中
    for k in range(2, 100):     # 考虑簇外，至少2个簇
        # 3.1 创建KMeans对象，参1：簇个数（k值），参2：最大迭代次数，参3：随机种子
        estimator = KMeans(n_clusters = k, max_iter = 100,random_state = 23)
        # 3.2 训练
        y_pred = estimator.fit(x)
        # 3.3 预测
        y_pred = estimator.predict(x)
        # 3.4 计算ch值（评估）,并添加到列表中,ch越小越好
        ch_list.append(calinski_harabasz_score(x, y_pred))

    # 4. 绘制ch曲线
    # 4.1 创建画布，指定画布尺寸
    plt.figure(figsize = (20,10))
    # 4.2 设置标题
    plt.title('ch')
    # 4.3 设置x刻度
    plt.xticks(range(0, 100, 3))
    # 4.4 添加网格
    plt.grid()
    # 4.5 添加x轴，y轴标签
    plt.xlabel('K')
    plt.ylabel('ch')
    # 4.6绘制折现图
    # 参1：x轴（k值），参2：y轴（ch）
    plt.plot(range(2, 100), ch_list)
    plt.show()

# 4. 测试
if __name__ == '__main__':
    # dm01_sse()
    # dm02_sc()
    dm03_ch()