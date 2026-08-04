# 导包
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# 数据预处理
# 读取数据
data = pd.read_csv('./data/churn.csv')
# print(data.head())
# data.info()
# one-hot热编码处理Churn与gender
data = pd.get_dummies(data, columns = ['Churn', 'gender'])
# print(data.head())
# data.info()
# 删掉多余列
data.drop(columns = ['Churn_No', 'gender_Female'], axis = 1, inplace = True)
# 修改列名
data.rename(columns = {'Churn_Yes': 'flag'}, inplace = True)
print(data.head())
data.info()
# 查看数据分布是否均匀
print(data.flag.value_counts())

# 数据可视化，绘制计数柱状图
sns.countplot(data, x = 'Contract_Month', hue = 'flag')
plt.show()

# 特征工程
x = data[['Contract_Month', 'internet_other', 'PaymentElectronic']]
y = data.flag
# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 888)
# 创建模型对象
estimator = LogisticRegression()
# 训练
estimator.fit(x_train, y_train)
# 预测
y_pre = estimator.predict(x_test)
print(y_pre)
# 评估
print(roc_auc_score(y_test, y_pre))
print(classification_report(y_test, y_pre))