# 导包
import pandas as pd
from sklearn.preprocessing import LabelEncoder          # 标签编码器
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier         # 集成学习
from sklearn.metrics import accuracy_score

# 1. 获取数据集
data = pd.read_csv('./data/wine0501.csv')
# data.info()
# print(data['Class label'].unique())       #有三种葡萄酒，[1 2 3]，但决策树只能用二叉树

# 2. 数据预处理
# 2.1 去掉红酒类别1
data = data[data['Class label'] != 1]
# print(data['Class label'].unique())         # [2 3]

# 3. 特征工程
# 3.1 特征提取
x = data[['Alcohol', 'Hue']]        # 酒精与色泽
y = data['Class label']
# 3.2 打印数据
# print(x.head())
# print(y.head())
# 2.4 通过 标签编码器，把 标签列，转换为 数值列.把标签[2 3]转为[0 1]
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)
# 2.5 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 23, stratify = y)

# 4. 模型训练,预测，评估
# 场景1：单一决策树——>充当弱分类器
# 4.1 创建模型对象
estimator1 = DecisionTreeClassifier(max_depth = 3)
# 4.2 训练模型
estimator1.fit(x_train, y_train)
# 4.2 模型预测
y_pred1 = estimator1.predict(x_test)
print(f'单一决策树预测：{y_pred1}')
# 4.3 模型评估
print(f'单一决策树评估：{accuracy_score(y_test, y_pred1)}')

# 场景2：AdaBoost——>集成学习
# 4.1 创建模型对象
# 参1：弱分类器(决策树对象)，参2：弱分类器个数，参3：学习率，参4：集成算法
estimator2 = AdaBoostClassifier(estimator1, n_estimators = 200, learning_rate = 0.1, algorithm = 'deprecated')
# 4.2 训练模型
estimator2.fit(x_train, y_train)
# 4.2 模型预测
y_pred2 = estimator2.predict(x_test)
print(f'AdaBoost集成学习预测：{y_pred2}')
# 4.3 模型评估
print(f'AdaBoost集成学习评估：{accuracy_score(y_test, y_pred2)}')
