# 导包
from sklearn.preprocessing import MinMaxScaler

# 1.准备数据集（归一化之前源数据）
x_train = [[90, 2, 10, 40],
           [60, 4, 15, 45],
           [75, 3, 13, 46]]

# 2. 创建归一化对象
# 参数feature_range，表示生成范围，默认（0，1）
transfor = MinMaxScaler(feature_range=(0, 1))

# 3. 对源数据集进行归一化处理
x_train_new = transfor.fit_transform(x_train)


# 打印处理后的数据
print(x_train_new)