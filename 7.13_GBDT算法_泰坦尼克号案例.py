# 导包
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

# 1. 加载数据
data = pd.read_csv('./data/train.csv')

# 2. 数据预处理
# 2.1 填充缺失值
data = data.copy()
data['Age'] = data['Age'].fillna(data.Age.mean())
# 2.2 将str类型变为bool类型
data = pd.get_dummies(data, columns = ['Sex'])
# data.info()

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
print(f'单个决策树预测结果：{y_pred1}')
# 4. 模型评估
# print(f'单个决策树模型评估报告：\n{classification_report(y_test, y_pred1)}')
print(f'单个决策树准确率：{accuracy_score(y_test, y_pred1)}')
print('-' * 50)

# 场景2：梯度提升分类树对象（GBDT），采用默认参数
# 1. 创建梯度提升分类树对象
estimator2 = GradientBoostingClassifier()
# 2. 模型训练
estimator2.fit(x_train, y_train)
# 3. 模型预测
y_pred2 = estimator2.predict(x_test)
print(f'梯度提升分类树预测结果:{y_pred2}')
# 4. 模型评估
print(f'梯度提升分类树准确率:{accuracy_score(y_test, y_pred2)}')
print('-' * 50)

# 场景3：梯度提升分类树对象（GBDT），采用网格搜索找超参数
# 1. 创建梯度提升分类树对象
estimator3 = GradientBoostingClassifier()
# 2. 参数准备
params = {
    'n_estimators': [100, 110],           # 弱学习器的数量
    'learning_rate': [0.1, 0.2],        # 学习率
    'max_depth': [3, 5],                # 树最大深度
}
# 3. 创建网格搜索对象,结合交叉验证
gs_estimator = GridSearchCV(estimator3, params, cv=3)
# 4. 网格搜索模型训练
gs_estimator.fit(x_train, y_train)
print(f"最优评估器: {gs_estimator.best_estimator_}")
# 5. 模型预测
y_pred3 = gs_estimator.predict(x_test)
print(f'梯度提升分类树预测结果:{y_pred3}')
# 4. 模型评估
print(f'梯度提升分类树准确率:{accuracy_score(y_test, y_pred3)}')
print('-' * 50)