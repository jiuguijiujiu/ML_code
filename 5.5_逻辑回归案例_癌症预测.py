# 导包
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. 加载数据
data = pd.read_csv('./data/breast-cancer-wisconsin.csv')
# data.info()                     # 产看数据信息

# 2. 数据预处理
# 2.1 把 异常值(?) 替换为 空值(np.nan)
# 参数1：要被替换的值    参数2：用来替换的值      参数3：是否修改源数据，默认False
data.replace(to_replace = '?', value = np.nan, inplace = True)
# 2.2 缺失值处理 ——> 删除
# 参数1：按行删还是按列删，默认0，按行删除
data.dropna(axis = 0, inplace = True)
# 2.3 打印处理后的信息
# data.info()

# 3. 特征工程（提取，预处理...）
# 3.1 特征提取，提取特征和标签
x = data.iloc[:, 1:-1]
# y = data.iloc[:, -1]
# y = data['class']         # 效果同上，因为标签列就是最后一列，列名为‘Class’
y = data.Class              # 效果同上
# 3.2 查看 特征 和 标签
# print(x.head())
print(x[:5])    # 效果同上
# print(y.head())
print(y[:5])    # 效果同上
print(x.shape, y.shape)
# 3.3 切割测试集与训练集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 666)
# 3.4 特征预处理：标准化
# 3.4.1 创建标准化对象
transfor = StandardScaler()
# 3.4.2 标准化训练集的特征，训练 + 标准化
x_train = transfor.fit_transform(x_train)
# 3.4.2 标准化测试集的特征，标准化
x_test = transfor.transform(x_test)


# 4. 模型训练
# 4.1 创建逻辑回归模型对象
estimator = LogisticRegression()
# 4.2 模型训练
estimator.fit(x_train, y_train)

# 5. 模型预测
y_pre = estimator.predict(x_test)
print(f'预测结果：{y_pre}')

# 6. 模型评估
# 正确率(准确率)，公式：预测对的/样本总数
print(f'预测前评估，正确率：{estimator.score(x_test, y_test)}')       # 测试集特征，测试集标签
print(f'预测后评估，正确率：{accuracy_score(y_test, y_pre)}')         # 测试集标签，预测值标签

# 思考：逻辑回归模型能用 准确率来评测吗？
# 答案：可以，但是结果不精准，因为逻辑回归模型主要用于 二分类
# 所以要通过 混淆矩阵来评测，即：精确率，召回率，F1值(F1-Score)，ROC曲线，AUC值。