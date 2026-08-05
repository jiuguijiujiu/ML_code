import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 提取数据
data = pd.read_csv('./data/train.csv')
# data.info()

# 数据预处理
data['Age'].fillna(data['Age'].mean(),inplace = True)
data = pd.get_dummies(data, columns = ['Sex'])
data.drop(columns = 'Sex_female', inplace = True)
# data.info()

# 特征工程
x = data[['Pclass', 'Age', 'Sex_male']]
y = data['Survived']
x.info()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# 模型训练
estimator = DecisionTreeClassifier(max_depth = 10)
estimator.fit(x_train, y_train)

# 模型预测
y_pre = estimator.predict(x_test)
print(y_pre)

# 模型评估
print(classification_report(y_test, y_pre))

# 绘制
plt.figure(figsize = (30, 30))
plot_tree(estimator, max_depth = 10, filled = True)
plt.show()