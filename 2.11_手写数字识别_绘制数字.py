# 导包
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib                       # 保存模型
from collections import Counter     # 去重统计

# 拓展：忽略警告
import warnings
warnings.filterwarnings('ignore', module = 'sklearn')       # 参数1：忽略警告  参数2：忽略的模块

# 1. 定义函数, 接收用户传入索引, 展示该索引对应图片
def show_digit(idx):
    # 1. 读取数据集,获取源数据
    df = pd.read_csv('./data/手写数字识别.csv')
    # print(df)       # [42000 rows x 785 columns]

    # 2. 判断传入的索引是否越界
    if idx < 0 or idx > len(df) - 1:
        return

    # 3.走到这里没有越界,正常获取数据
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    # 4. 查看用户传入的索引图片 --> 是几
    print(f'该图片对应数字是{y.iloc[idx]}')
    print(f'查看所有标签分布情况：{Counter(y)}')

    # 5. 查看 用户传入的索引图片 的形状
    print(x.iloc[idx].shape)        # 我们要把(784,) 变成 (28,28)
    # print(x.iloc[idx].values)         # 具体的784个像素点数据

    # 6. 把(784,) 变成 (28,28)
    x = x.iloc[idx].values.reshape(28, 28)
    # print(x)          # 28*28像素点

    # 7. 具体的绘制灰度图动作
    plt.imshow(x, cmap='gray')      # gray灰度图
    plt.axis('off')                 # 关闭坐标
    plt.show()

# 2.定义函数，训练模型，并保持训练好的模型
def train_model():
    # 1. 导入数据
    df = pd.read_csv('./data/手写数字识别.csv')

    # 2. 数据预处理，划分数据集与测试集
    # 2.1 拆分出特征列与标签列
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    # 2.2 打印特征列与标签列形状
    print(f'特征列形状：{x.shape}')           # (42000, 784)
    print(f'标签列列形状：{y.shape}')          # (42000,)
    print(f'查看所有标签分布情况：{Counter(y)}')
    # 2.3 对特征列左归一化
    x = x / 255
    # 2.4 划分数据集与测试集
    # 参数5 参考 y中分布比例 进行抽取，保持数据均衡
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 23, stratify = y)

    # 3. 模型训练
    # 3.1 创建模型对象
    estimator = KNeighborsClassifier(n_neighbors=3)
    # 3.2 模型训练动作
    estimator.fit(x_train, y_train)

    # 4. 模型评估
    print(f'准确率：{estimator.score(x_test, y_test)}')
    print(f'准确率：{accuracy_score(y_test, estimator.predict(x_test))}')

    # 保存模型
    # 参数1：模型对象      参数2：保存路径
    joblib.dump(estimator, './model/knn.pkl')           # pickle文件：python（pandas）独有的文件类型
    print("模型保存成功")

# 3. 定义函数，测试模型
def use_model():
    # 1. 加载图片
    img = plt.imread('./data/demo.png')        # 28 * 28 像素

    # 2. 绘制图片
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # 3. 加载模型
    estimator = joblib.load('./model/knn.pkl')

    # 4. 模型预测
    # 4.1 将28 * 28 转为 1 * 784
    print(f'{img.shape}')
    # print(f'{img.reshape(1, 784).shape}')
    print(f'{img.reshape(1, -1).shape}')            # 语法糖，效果同上
    # 4.2 具体 拿图片 转换成 结果，记得归一化
    # img = img.reshape(1, -1) / 255          # 可能会预测失败，因为读图的时候，像素值可能不是特别的精准
    img = img.reshape(1, -1)                # 用原始的读取到的像素值，做预测，它本身就是小于1的小数，不是0-255，不要归一化
    # 4.3 模型预测
    y_pre = estimator.predict(img)
    # 4.4 打印预测结果
    print(f'预测结果：{y_pre}')

# 4. 测试
if __name__ == "__main__":
    # show_digit(9)
    # train_model()
    # train_model()
    use_model()