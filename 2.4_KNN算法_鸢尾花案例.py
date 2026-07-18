# 导包
from sklearn.datasets import load_iris                      # 加载鸢尾花测试集的I
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split        # 分割训练集和测试集的
from sklearn.preprocessing import StandardScaler            # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier          # KNN算法 分类对象
from sklearn.metrics import accuracy_score                  # 模型评估的，计算模型预测的准确率

# 1. 定义函数，加载鸢尾花数据集，并查看数据集
def dm01_load_iris():
    # 1. 加载数据集
    iris_data = load_iris()
    # 2. 查看数据集
    # print(f"数据集：{iris_data}")               # 字典形态
    # print(f"数据集类型：{type(iris_data)}")       # <class 'sklearn.utils._bunch.Bunch'>
    # 3. 查看数据集所有的键
    # print(f"数据集所有的键：{iris_data.keys()}")
    # 查看数据集键对应的值
    print(f"具体的数据：{iris_data.data[:5]}")            # 数据有150条，每个数据有4个特征，我们只看前5条
    print(f"具体的标签：{iris_data.target[:5]}")          # 数据有150条，每个数据有1个表，我们只看前5条
    print(f"标签对应的名称：{iris_data.target_names}")      # ['setosa' 'versicolor' 'virginica']
    print(f"特征对应的名称：{iris_data.feature_names}")     # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    # print(f"数据集的描述信息：{iris_data.DESCR}")
    # print(f"数据集的框架：{iris_data.frame}")                        # None
    # print(f"数据集的文件名：{iris_data.filename}")                    # iris.csv
    # print(f"数据集的模型（在哪个包下面）：{iris_data.data_module}")      # sklearn.datasets.data

# 2. 定义函数，绘制数据集散点图（可视化）
def dm02_show_iris():
    # 1. 加载数据集
    iris_data = load_iris()
    # 2. 将鸢尾花数据集 封装成 dataframe对象
    iris_df = pd.DataFrame(data = iris_data.data, columns = iris_data.feature_names)
    # 3. 给df对象新增一列 ——> 标签列
    iris_df['label'] = iris_data.target
    # print(iris_df)
    # 4. 通过 seaborn 绘制散点图
    # 参数1：数据集，参数2：x轴，参数3：y轴，参数4：分组字段，参数5：是否显示拟合回归线
    sns.lmplot(data = iris_df, x = 'sepal length (cm)', y = 'sepal width (cm)', hue = 'label', fit_reg = True)
    # 设置标题显示
    plt.title("iris data")
    plt.xlabel("sepal length (cm)")
    plt.ylabel("sepal width (cm)")
    plt.tight_layout()              # 自动调整子图参数，以使整个图像的边界与子图匹配
    plt.show()

# 3. 定义函数，切分训练集与测试集
def dm03_split_train_test():
    # 1. 加载数据集
    iris_data = load_iris()
    # 2. 数据的预处理：8:2的比例 切分数据集
    # 参数1:特征数据    参数2：标签数据  参数3：测试集比例  参数4：随机种子
    # 返回值：训练集特征，测试集特征，训练集标签，测试集标签
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size = 0.2, random_state = 23)      # 返回元组
    # 3. 打印切割后的结果
    print(f"训练集特征：{x_train}, 个数：{len(x_train)}")        # 120条，每条4列
    print(f"训练集标签：{y_train}, 个数：{len(y_train)}")        # 120条，每条1列
    print(f"测试集特征：{x_test}, 个数：{len(y_test)}")          # 30条，每条4列
    print(f"测试集标签：{y_test}, 个数：{len(y_test)}")          # 30条，每条1列

# 4. 定义函数，完成鸢尾花完整案例：加载数据，数据预处理，特征工程，模型训练，模型评估
def dm04_evaluate_test():
    # 1. 加载数据
    iris_data = load_iris()

    # 2. 数据预处理:以8:2的比例 切分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size = 0.2, random_state = 23)

    # 3. 特征工程(提取,预处理...)
    # 思考1: 特征提取: 因为数据集中,4个数据我们都需要,所以不需要进行特征提取
    # 思考2: 特征预处理: 数据集中,数据特征间 差值不多大,所以无需进行预处理.但是加入后能使代码更加完善
    # 3.1 创建 标准化对象
    transfor = StandardScaler()
    # 3.2 对 特征数据 进行标准化
    # fit_transform: 兼具fit和transform的功能，即：训练、转换。该函数适用于：第一次进行标准化的时候使用。一般用于处理：训练集。
    x_train = transfor.fit_transform(x_train)
    # transform: 只有转换。该函数适用于：重复进行标准化动作时使用，一般用于对测试集进行标准化。
    x_test = transfor.transform(x_test)

    # 4. 模型训练
    # 4.1 创建模型训练对象
    estimator = KNeighborsClassifier(n_neighbors = 3)
    # 4.2 具体的训练动作
    estimator.fit(x_train, y_train)

    # 5. 模型预测
    # 场景1:对刚才切分的30条测试集进行预测
    y_pre = estimator.predict(x_test)
    print(f"预测结果为{y_pre}")

    # 场景2:对150条以外的新数据集进行预测
    # 自己创建一个新的数据集
    my_data = [[7.8, 2.1, 3.9, 1.6]]
    # 数据标准化
    my_data = transfor.transform(my_data)
    # 模型预测
    my_data_pre = estimator.predict(my_data)
    print(f"自己数据集预测结果为{my_data_pre}")

    # 5.1 查看上述数据集,每种分类的预测概率
    my_data_pre_proba = estimator.predict_proba(my_data)
    print(f"预测概率为:{my_data_pre_proba}")

    # 6. 模型评估
    # 方式1: 直接评分:基于 测试集特征 和 测试集标签
    print(f"准确率(正确率): {estimator.score(x_test, y_test)}")     # 0.9666666666666667

    # 方式2: 基于 测试集标签 与 测试集预测结果 进行评分
    print(f"准确率(正确率): {accuracy_score(y_test, y_pre)}")      # 0.9666666666666667



# 5. 测试
if __name__ == '__main__':
    # dm01_load_iris()
    # dm02_show_iris()
    # dm03_split_train_test()
    dm04_evaluate_test()

