# 导包
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 1. 加载数据
data = pd.read_csv('./data/train.csv')

# 2. 数据预处理
# 2.1 填充缺失值
data = data.copy()
data['Age'] = data['Age'].fillna(data.Age.mean())
# 2.2 将str类型变为bool类型
data = pd.get_dummies(data, columns = ['Sex'])
data.info()

# 3. 特征工程
# 3.1 特征选取
x = data[['Pclass', 'Sex_male', 'Age']]
y = data.Survived
# 3.2 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# 4. 模型训练,预测，评估
# 场景1：单一决策树
# 1. 创建决策树模型对象
estimator1 = DecisionTreeClassifier()
# 2. 模型训练
estimator1.fit(x_train, y_train)
# 3. 模型预测
y_pred1 = estimator1.predict(x_test)
print(f'决策树预测结果{y_pred1}')
# 4. 模型评估
print(f'决策树准确率{estimator1.score(x_test, y_test)}')
print('-' * 50)

# 场景2：随机森林算法，采用默认参数
# 1. 创建随机森林模型对象
estimator2 = RandomForestClassifier()
# 2. 模型训练
estimator2.fit(x_train, y_train)
# 3. 模型预测
y_pred2 = estimator2.predict(x_test)
print(f'随机森林预测结果{y_pred2}')
# 4. 模型评估
print(f'随机森林准确率{estimator2.score(x_test, y_test)}')
print('-' * 50)

# 场景3：随机森林算法，采用网格搜索找超参数
# 1. 创建随机森林模型对象
estimator3 = RandomForestClassifier()       # n_estimators = 100, max_depth = None
# 2. 参数准备
params = {'n_estimators':[30, 50, 70, 90], 'max_depth':[3, 5, 7, 9]}
# 3. 创建网格搜索对象,结合交叉验证
gs_estimator = GridSearchCV(estimator3, params, cv=3)
# 4. 网格搜索模型训练
gs_estimator.fit(x_train, y_train)
print(f"最优评估器: {gs_estimator.best_estimator_}")
# 5. 模型预测
y_pred3 = gs_estimator.predict(x_test)
print(f'随机森林预测结果{y_pred3}')
# 4. 模型评估
print(f'随机森林准确率{gs_estimator.score(x_test, y_test)}')
print('-' * 50)