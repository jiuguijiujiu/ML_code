# 导包
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, classification_report


# 1. 定义函数，演示：数据的预处理.
def dm01_data_preprocess():
    # 1. 读取数据
    churn_df = pd.read_csv('./data/churn.csv')
    # 2. 查看（处理前）数据集
    # churn_df.info()
    # print(churn_df.head())
    # 3. 因为Churn 和 gender列是字符串，所以需要进行one-hot编码(热编码处理).
    churn_df = pd.get_dummies(churn_df, columns = ['Churn', 'gender'])
    # 4. 查看（处理后）的数据集
    # churn_df.info()
    # print(churn_df.head())
    # 5. 删除one-hot后，新增的冗余列
    # 参1：要删除的列  参2：axis = 1（删除列），默认0（删除行）  参3：是否修改源数据
    churn_df.drop(['Churn_No', 'gender_Female'], axis = 1, inplace = True)
    # churn_df.info()
    # print(churn_df.head())
    # 6. 修改列名，将Churn_Yes ——> flag，充当标签列
    churn_df.rename(columns = {'Churn_Yes': 'flag'}, inplace = True)
    churn_df.info()
    print(churn_df.head())      # ture ——> 流失，false ——> 不流失
    # 7. 查看数据集的分布
    print(churn_df.flag.value_counts())         # False：5174   True：1869

# 2. 定义函数，演示：数据的可视化.
def dm02_data_visualization():
    # 1. 读取数据
    churn_df = pd.read_csv('./data/churn.csv')
    # 2. 因为Churn 和 gender列是字符串，所以需要进行one-hot编码(热编码处理).
    churn_df = pd.get_dummies(churn_df, columns = ['Churn', 'gender'])
    # 3. 删除one-hot后，新增的冗余列
    # 参1：要删除的列  参2：axis = 1（删除列），默认0（删除行）  参3：是否修改源数据
    churn_df.drop(['Churn_No', 'gender_Female'], axis = 1, inplace = True)
    # 4. 修改列名，将Churn_Yes ——> flag，充当标签列
    churn_df.rename(columns = {'Churn_Yes': 'flag'}, inplace = True)
    churn_df.info()
    print(churn_df.head())      # ture ——> 流失，false ——> 不流失
    # 5. 查看数据集的分布
    print(churn_df.flag.value_counts())         # False：5174   True：1869
    # 6. 产看列名，方便后续的特征提取
    print(churn_df.columns)
    # ['Partner_att', 'Dependents_att', 'landline', 'internet_att',
    #  'internet_other', 'StreamingTV', 'StreamingMovies', 'Contract_Month',
    #  'Contract_1YR', 'PaymentBank', 'PaymentCreditcard', 'PaymentElectronic',
    #  'MonthlyCharges', 'TotalCharges', 'flag', 'gender_Male']

    # 7. 数据可视化，绘制计数柱状图
    # 参1：数据集  参2：x轴（月度会员）  参3：分组字段，flag（是否流失，ture ——> 流失，false ——> 不流失）
    sns.countplot(data = churn_df, x = 'Contract_Month', hue = 'flag')
    plt.show()

# 3. 定义函数，演示：逻辑回归算法的模型训练,预测,评估.
def dm03_logistic_regression():
    # 1. 加载数据集
    churn_df = pd.read_csv('./data/churn.csv')

    # 2. 数据预处理
    # 2.1 因为Churn 和 gender列是字符串，所以需要进行one-hot编码(热编码处理).
    churn_df = pd.get_dummies(churn_df, columns=['Churn', 'gender'])
    # 2.2 删除one-hot后，新增的冗余列
    # 参1：要删除的列  参2：axis = 1（删除列），默认0（删除行）  参3：是否修改源数据
    churn_df.drop(['Churn_No', 'gender_Female'], axis=1, inplace=True)
    # 2.3 修改列名，将Churn_Yes ——> flag，充当标签列
    churn_df.rename(columns={'Churn_Yes': 'flag'}, inplace=True)

    # 3. 特征工程（提取，预处理，选择，降维，组合）
    # 3.1 提取特征列与标签列
    # x的特征列： 月度会员，是否有互联网服务，是否是电子支付
    x = churn_df[['Contract_Month', 'internet_other', 'PaymentElectronic']]
    y = churn_df['flag']            # flag（是否流失，ture ——> 流失，false ——> 不流失）
    # 3.2 划分数据集与训练集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 888, stratify = y)

    # 4. 模型训练
    # 4.1 创建模型对象
    estimator = LogisticRegression()
    # 4.2 模型训练
    estimator.fit(x_train, y_train)

    # 5.  模型预测
    y_pre = estimator.predict(x_test)
    print(f'预测结果：{y_pre}')

    # 6.  模型评估
    print(f'准确率：{estimator.score(x_test, y_test)}')
    print(f'准确率：{accuracy_score(y_test, y_pre)}')
    cm = confusion_matrix(y_test, y_pre)
    df_label = ['流失（正例）', '不流失（反例）']
    cm_df = pd.DataFrame(cm, index = df_label, columns = df_label)
    print(f'混淆矩阵：{cm_df}')
    print(f'精确率：{precision_score(y_test, y_pre)}')
    print(f'召回率：{recall_score(y_test, y_pre)}')
    print(f'f1：{f1_score(y_test, y_pre)}')

    print(f'roc_auc_score：{roc_auc_score(y_test, y_pre)}')
    print(f'分类评估报告：{classification_report(y_test, y_pre)}')

# 4. 测试
if __name__ == '__main__':
    # dm01_data_preprocess()
    # dm02_data_visualization()
    dm03_logistic_regression()