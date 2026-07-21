# 导包
from sklearn.datasets import load_iris                                      # 加载鸢尾花测试集的I
from sklearn.model_selection import train_test_split, GridSearchCV          # 分割训练集和测试集的, 寻找最优超参数(网格搜索+交叉校验)
from sklearn.preprocessing import StandardScaler                            # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier                          # KNN算法 分类对象
from sklearn.metrics import accuracy_score                                  # 模型评估的，计算模型预测的准确率

# 1. 加载鸢尾花数据集
iris_data =load_iris()

# 2. 数据预处理: 以8:2切分训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size = 0.2, random_state = 23)

# 3. 特征工程(提取,预处理...)
# 标准化
# 3.1 创建标准化对象
transfor = StandardScaler()
# 3.1 标准化
x_train = transfor.fit_transform(x_train)
x_test = transfor.fit_transform(x_test)

# 4. 模型训练
# 4.1 创建分类模型对象
estimator = KNeighborsClassifier()
# 4.2 定义字典,记录超参可能出现的值
param_dict = {'n_neighbors':[i for i in range(1,11)]}
# 4.3 创建GridSearchCV 对象 --> 寻找最优超参数(网格搜索+交叉校验)
# 参数1: 要计算超参数的 模型对象
# 参数2: 超参数可能出现的情况
# 参数3: 交叉验证的折数, 这里是4折校验: 每个超参数进行 4次交叉校验, 4*10 = 40次
# 返回值 estimator 处理后的模型对象
estimator = GridSearchCV(estimator = estimator, param_grid = param_dict, cv = 4)        # 怎么感觉像是在原有模型上 添加了功能
# 4.4 模型训练
estimator.fit(x_train, y_train)
# 4.5 打印最优超参数组合
print(f"最优评分: {estimator.best_score_}")                 # 0.9583333333333334
print(f"最优超参数组合: {estimator.best_params_}")           # {'n_neighbors': 7}
print(f"最优评估器: {estimator.best_estimator_}")           # KNeighborsClassifier(n_neighbors=7)
print(f"具体交叉验证结果: {estimator.cv_results_}")

# 5. 模型评估
# 5.1 获取最优模型对象
# estimator = estimator.best_estimator_
estimator = KNeighborsClassifier(n_neighbors=7)
# 5.2 模型训练
estimator.fit(x_train, y_train)
# 5.3 模型评估
y_pred = estimator.predict(x_test)
# 5.4 模型预测
print(f"准确率: {accuracy_score(y_test, y_pred)}")



