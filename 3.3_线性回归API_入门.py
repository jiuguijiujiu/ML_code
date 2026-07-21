# 导包
from sklearn.linear_model import LinearRegression

# 1. 准备数据
x_train = [[160], [166], [172], [174], [180]]           # 训练集特征
y_train = [56.3, 60.6, 65.1, 68.5, 75]                  # 训练集标签
x_test = [[176]]                                        # 测试集特征


# 2. 数据预处理，这里不需要
# 3. 特征工程（特征提取，特征预处理），这里不需要

# 4. 模型训练
# 4.1 创建模型对象
estimator = LinearRegression()
# 4.2 具体的训练动作
estimator.fit(x_train, y_train)
# 4.3 因为是线性回归模型，我们可以查看下：斜率(w，权重)，截距(b，偏置)
print(f'权重：{estimator.coef_}')             # [0.92942177]
print(f'偏执：{estimator.intercept_}')        # -93.27346938775517


# 5. 模型预测
y_pre = estimator.predict(x_test)
print(y_pre)            # [70.3047619]

# 6. 模型评估，还没学，先放着