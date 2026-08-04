# 导包
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1. 加载数据
data = pd.read_csv('./data/train.csv')
# print(data.head())
# data.info()

# 2. 数据预处理
# 2.1 提取特征于标签
x = data[['Pclass', 'Sex', 'Age']]
y = data.Survived
# print(x.head())
# print(y.head())
# x.info()

# 2.2 发现Age列有确实，我们用该列的 平均值做填充.
# x['Age'].fillna(x['Age'].mean(), inplace=True)          #会报警告，但可以用
# x['Age'] = x['Age'].fillna(x['Age'].mean())             # 还是报警告，因为直接修改源数据
# 解决方案，copy()数据之后再改
x = x.copy()                                              # 拷贝数据
x['Age'] = x['Age'].fillna(x['Age'].mean())
# x.info()

# 2.3针对于Sex列做one—hot（热编码）
x = pd.get_dummies(x, columns = ['Sex'])

# 2.4 查看处理后的数据集.
# x.info()

# 2.5 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 888)

# 3. 特征工程

# 4. 模型训练
# 参1：决策树最多10层
estimator = DecisionTreeClassifier(max_depth = 10)
estimator.fit(x_train, y_train)

# 5. 模型预测
y_pre = estimator.predict(x_test)
print(y_pre)

# 6. 模型评估
print(classification_report(y_test, y_pre))

# 7. 绘制决策树图
# 参1：设置画布（图片）大小, 30 * 100(dpi) * 20 * 100(dpi) = 3000 * 2000像素
plt.figure(figsize = (30, 20))
# 参1：模型对象 参2：是否颜色填充 参3：树最大深度
plot_tree(estimator, filled = True,max_depth = 10)
plt.savefig('./data/tree.png')
plt.show()