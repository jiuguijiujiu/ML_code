# 导包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error          # 计算均方误差
from sklearn.model_selection import train_test_split

# 1. 定义函数，模拟欠拟合
def dm01_under_fitting():
    # 1. 准备数据
    # 1.1 设置随机数种子
    np.random.seed(888)     # 设置随机数种子，让每次随机的数不变
    # 1.2 生成一个特征(x)：100个
    # 参数1： 最小值
    # 参数2： 最大值
    # 参数3： size，生成个数
    x = np.random.uniform(-3,3, size=100)
    # 1.3 生成一个标签(y)：100个
    # np.random.normal(0,1,size=100)，生成100个均值为0，标准差为1的100个正态分布数，充当噪点
    # 线性公式：y = kx + b
    y = 0.5*x**2 + x + 2 + np.random.normal(0,1,size=100)
    # 1.4 查看随机生成的特征（x），标签（y）数据
    # print(f'特征(x)：{x}')       # [1, 2, 3, 4, 5]
    # print(f'标签(y)：{y}')

    # 2. 数据预处理
    # 2.1 将特征x的一行多列，转为多行一列
    X = x.reshape(-1,1)
    # print(f'处理后的特征：{X}')      # [[1], [2], [3], [4], [5]]

    # 3. 特征工程（因为特征范围是-3~3,范围小，所以不用标注化）
    # 3.1 切割数据(这里不做了，直接用100个数据训练，100个数据测试)
    # x_train, x_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=888)

    # 4. 模型训练
    # 4.1 创建模型对象
    estimator = LinearRegression()      # 正规方程 线性回归 模型
    # 4.2 模型训练
    estimator.fit(X, y)

    # 5. 模型预测
    y_pre = estimator.predict(X)

    # 6. 模型评估
    # 6.1 均方差评估
    print(f'均方差：{mean_squared_error(y, y_pre)}')

    # 7. 绘图
    plt.scatter(x,y)                             # 散点图，展示真实值
    plt.plot(x,y_pre, color = 'red')       # 折线图，展示预测值
    plt.show()

# 2. 定义函数，模拟正好拟合
def dm02_just_fitting():
    # 1. 准备数据
    # 1.1 设置随机数种子
    np.random.seed(888)     # 设置随机数种子，让每次随机的数不变
    # 1.2 生成一个特征(x)：100个
    # 参数1： 最小值
    # 参数2： 最大值
    # 参数3： size，生成个数
    x = np.random.uniform(-3,3, size=100)
    # 1.3 生成一个标签(y)：100个
    # np.random.normal(0,1,size=100)，生成100个均值为0，标准差为1的100个正态分布数，充当噪点
    # 线性公式：y = kx + b
    y = 0.5*x**2 + x + 2 + np.random.normal(0,1,size=100)
    # 1.4 查看随机生成的特征（x），标签（y）数据
    print(f'特征(x)：{x[:5]}')       # [1, 2, 3, 4, 5]
    print(f'标签(y)：{y[:4]}')

    # 2. 数据预处理
    # 2.1 将特征x的一行多列，转为多行一列
    X = x.reshape(-1,1)             # [[1], [2], [3], [4], [5]]
    print(f'处理后的特征：{X[:5]}')      # [[1], [2], [3], [4], [5]]
    # 2.2 因为目前特征列只有1列，模型过于简单，会出现欠拟合的问题，我们增加1列 特征列，从而增加模型的复杂度。
    # 即：把数据从 [[1], [2], [3], [4], [5]] ⇒ [[1, 1], [2, 4], [3, 9], [4, 16], [5, 25]]
    X2 = np.hstack([X, X ** 2])       # 该函数作用：横向拼接，即：拼接2个数组，拼接后数组的行数不变，拼接后数组的列数等于拼接前数组的列数与新增列的个数之和。
    print(f'处理后的特征：{X2[:5]}')      #


    # 3. 特征工程（因为特征范围是-3~3,范围小，所以不用标注化）
    # 3.1 切割数据(这里不做了，直接用100个数据训练，100个数据测试)
    # x_train, x_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=888)

    # 4. 模型训练
    # 4.1 创建模型对象
    estimator = LinearRegression()      # 正规方程 线性回归 模型
    # 4.2 模型训练
    estimator.fit(X2, y)

    # 5. 模型预测
    y_pre = estimator.predict(X2)

    # 6. 模型评估
    # 6.1 均方差评估
    print(f'均方差：{mean_squared_error(y, y_pre)}')

    # 7. 绘图
    plt.scatter(x,y)  # 散点图，展示真实值
    # np.sort(x): 对x轴(特征)排序，默认是：升序。
    # np.argsort(x): 对x轴(特征)排序，返回排序后的索引。
    # 例如： 排序前x轴是 [11, 33, 22] → 对应索引: [0, 1, 2]
    # 排序后：x轴是 [11, 22, 33] → 对应索引: [0, 2, 1]
    plt.plot(np.sort(x) ,y_pre[np.argsort(x)], color = 'red')       # 折线图，展示预测值
    plt.show()


# 3. 定义函数，模拟过拟合
def dm03_over_fitting():
    # 1. 准备数据
    # 1.1 设置随机数种子
    np.random.seed(888)     # 设置随机数种子，让每次随机的数不变
    # 1.2 生成一个特征(x)：100个
    # 参数1： 最小值
    # 参数2： 最大值
    # 参数3： size，生成个数
    x = np.random.uniform(-3,3, size=100)
    # 1.3 生成一个标签(y)：100个
    # np.random.normal(0,1,size=100)，生成100个均值为0，标准差为1的100个正态分布数，充当噪点
    # 线性公式：y = kx + b
    y = 0.5*x**2 + x + 2 + np.random.normal(0,1,size=100)
    # 1.4 查看随机生成的特征（x），标签（y）数据
    print(f'特征(x)：{x[:5]}')       # [1, 2, 3, 4, 5]
    print(f'标签(y)：{y[:4]}')

    # 2. 数据预处理
    # 2.1 将特征x的一行多列，转为多行一列
    X = x.reshape(-1,1)             # [[1], [2], [3], [4], [5]]
    print(f'处理后的特征：{X[:5]}')      # [[1], [2], [3], [4], [5]]
    # 2.2 因为目前特征列只有1列，模型过于简单，为了模拟过拟合，我们增加9列 特征列，从而增加模型的复杂度。
    # 即：把数据从 [[1], [2], [3], [4], [5]] ⇒ [[1, 1**2, 1**3, 1**4, 1**5...], [2, 2**2, 2**3, 2**4, 2**5...], [3...]...]
    # 该函数作用：横向拼接，即：拼接多个数组，拼接后数组的行数不变，拼接后数组的列数等于拼接前数组的列数与新增列的个数之和。
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    print(f'处理后的特征：{X3[:5]}')


    # 3. 特征工程（因为特征范围是-3~3,范围小，所以不用标注化）
    # 3.1 切割数据(这里不做了，直接用100个数据训练，100个数据测试)
    # x_train, x_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=888)

    # 4. 模型训练
    # 4.1 创建模型对象
    estimator = LinearRegression()      # 正规方程 线性回归 模型
    # 4.2 模型训练
    estimator.fit(X3, y)

    # 5. 模型预测
    y_pre = estimator.predict(X3)

    # 6. 模型评估
    # 6.1 均方差评估
    print(f'均方差：{mean_squared_error(y, y_pre)}')

    # 7. 绘图
    plt.scatter(x,y)  # 散点图，展示真实值
    # np.sort(x): 对x轴(特征)排序，默认是：升序。
    # np.argsort(x): 对x轴(特征)排序，返回排序后的索引。
    # 例如： 排序前x轴是 [11, 33, 22] → 对应索引: [0, 1, 2]
    # 排序后：x轴是 [11, 22, 33] → 对应索引: [0, 2, 1]
    plt.plot(np.sort(x) ,y_pre[np.argsort(x)], color = 'red')       # 折线图，展示预测值
    plt.show()

# 4. 定义函数，演示：L1正则化
def dm04_L1_regularization():
    # 1. 准备数据
    # 1.1 设置随机数种子
    np.random.seed(888)     # 设置随机数种子，让每次随机的数不变
    # 1.2 生成一个特征(x)：100个
    # 参数1： 最小值
    # 参数2： 最大值
    # 参数3： size，生成个数
    x = np.random.uniform(-3,3, size=100)
    # 1.3 生成一个标签(y)：100个
    # np.random.normal(0,1,size=100)，生成100个均值为0，标准差为1的100个正态分布数，充当噪点
    # 线性公式：y = kx + b
    y = 0.5*x**2 + x + 2 + np.random.normal(0,1,size=100)
    # 1.4 查看随机生成的特征（x），标签（y）数据
    print(f'特征(x)：{x[:5]}')       # [1, 2, 3, 4, 5]
    print(f'标签(y)：{y[:4]}')

    # 2. 数据预处理
    # 2.1 将特征x的一行多列，转为多行一列
    X = x.reshape(-1,1)             # [[1], [2], [3], [4], [5]]
    print(f'处理后的特征：{X[:5]}')      # [[1], [2], [3], [4], [5]]
    # 2.2 因为目前特征列只有1列，模型过于简单，为了模拟过拟合，我们增加9列 特征列，从而增加模型的复杂度。
    # 即：把数据从 [[1], [2], [3], [4], [5]] ⇒ [[1, 1**2, 1**3, 1**4, 1**5...], [2, 2**2, 2**3, 2**4, 2**5...], [3...]...]
    # 该函数作用：横向拼接，即：拼接多个数组，拼接后数组的行数不变，拼接后数组的列数等于拼接前数组的列数与新增列的个数之和。
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    print(f'处理后的特征：{X3[:5]}')


    # 3. 特征工程（因为特征范围是-3~3,范围小，所以不用标注化）
    # 3.1 切割数据(这里不做了，直接用100个数据训练，100个数据测试)
    # x_train, x_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=888)

    # 4. 模型训练
    # 4.1 创建模型对象
    estimator = LinearRegression()      # 正规方程 线性回归 模型
    # 4.2 模型训练
    estimator.fit(X3, y)

    # 5. 模型预测
    y_pre = estimator.predict(X3)

    # 6. 模型评估
    # 6.1 均方差评估
    print(f'均方差：{mean_squared_error(y, y_pre)}')

    # 7. 绘图
    plt.scatter(x,y)  # 散点图，展示真实值
    # np.sort(x): 对x轴(特征)排序，默认是：升序。
    # np.argsort(x): 对x轴(特征)排序，返回排序后的索引。
    # 例如： 排序前x轴是 [11, 33, 22] → 对应索引: [0, 1, 2]
    # 排序后：x轴是 [11, 22, 33] → 对应索引: [0, 2, 1]
    plt.plot(np.sort(x) ,y_pre[np.argsort(x)], color = 'red')       # 折线图，展示预测值
    plt.show()

# 5.
# 6. 测试
if __name__ == '__main__':
    # dm01_under_fitting()
    # dm02_just_fitting()
    # dm03_over_fitting()
    dm04_L1_regularization()