# 导包
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib                       # 保存模型
from collections import Counter     # 去重统计

# 1. 定义函数, 接收用户传入索引, 展示该索引对应图片
def show_digit(idx):
    # 1. 读取数据集,获取源数据
    df = pd.read_csv('./data/手写数字识别.csv')
    # print(df)       # [42000 rows x 785 columns]

    # 2. 判断传入的索引是否越界
    if idx < 0 or idx > len(df) - 1:
        return

    # 3.走到这里没有越界,正常获取数据
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    # 4. 查看用户传入的索引图片 --> 是几
    print(f'该图片对应数字是{y.iloc[idx]}')

    # 5. 查看 用户传入的索引图片 的形状
    print(x.iloc[idx].shape)        # 我们要把(784,) 变成 (28,28)
    # print(x.iloc[idx].values)         # 具体的784个像素点数据

    # 6. 把(784,) 变成 (28,28)






# 2.

# 3.

# 4. 测试
if __name__ == "__main__":
    show_digit(9)