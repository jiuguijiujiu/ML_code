# 导包
from sklearn.preprocessing import StandardScaler

# 1. 准备数据集
x_train = [[90, 2, 10, 40],
           [60, 4, 15, 45],
           [75, 3, 13, 46]]

# 2. 创建标准化对象
transfor = StandardScaler()

# 3. 对源数据进行标准化处理
x_train_new = transfor.fit_transform(x_train)

# 打印处理后数据
print(f"标准化处理后的数据：{x_train_new}")

# 打印 各列 平均值 与 标准差，方差
print(f"平均值：{transfor.mean_}")
print(f"标准差：{transfor.scale_}")
print(f"方差：{transfor.var_}")