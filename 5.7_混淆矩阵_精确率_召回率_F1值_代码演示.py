# 导包
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score      # 混淆矩阵，精确率，召回率，f1值

# 需求：已知有10个样本，6个恶性肿瘤(正例)，4个良性肿瘤(反例).
# 模型A预测结果为： 预测对了3个恶性肿瘤， 预测对了4个良性肿瘤
# 模型B预测结果为： 预测对了6个恶性肿瘤， 预测对了1个良性肿瘤
# 请针对于上述的数据集，搭建 混淆矩阵，并分别计算模型A，模型B的 精确率，召回率，F1值.

# 1. 定义变量，记录：样本数据
y_train = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']

# 2. 定义变量，记录：模型A的预测结果
y_pred_A = ['恶性', '恶性', '恶性', '良性', '良性', '良性', '良性', '良性', '良性', '良性']

# 3. 定义变量，记录：模型B的预测结果
y_pred_B = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '恶性', '恶性', '恶性']

# 4. 用标签标记 正例，反例.
label = ['恶性', '良性']
df_label = ['恶性(正例)', '良性(反例)']

# 5. 针对于 真实值(y_train) 和 模型A的预测结果(y_pred_A)，搭建 混淆矩阵.
cm_A = confusion_matrix(y_train, y_pred_A)
print(f'混淆矩阵A：\n{cm_A}')

# 6. 为了测试结果更好看，把上述的 混淆矩阵 转换成 DataFrame.
df_A = pd.DataFrame(cm_A, index = df_label, columns = df_label)
print(f'混淆矩阵A：\n{df_A}')

# 7. 针对于 真实值(y_train) 和 模型B的预测结果(y_pred_B)，搭建 混淆矩阵.
cm_B = confusion_matrix(y_train, y_pred_B)
print(f'混淆矩阵B：\n{cm_B}')

# 8. 为了测试结果更好看，把上述的 混淆矩阵 转换成 DataFrame.
df_B = pd.DataFrame(cm_B, index = df_label, columns = df_label)
print(f'混淆矩阵A：\n{df_B}')

# 9. 计算A模型的 精确率，召回率，F1值.
print(f'A模型精确率：{precision_score(y_train, y_pred_A, pos_label = '恶性')}')     # 参1：真实值  参2：预测值  参3：正例标签
print(f'A模型召回率：{recall_score(y_train, y_pred_A, pos_label = '恶性')}')
print(f'A模型f1：{f1_score(y_train, y_pred_A, pos_label = '恶性')}')

# 10. 计算B的 精确率，召回率，F1值.
print(f'B模型精确率：{precision_score(y_train, y_pred_B, pos_label = '恶性')}')     # 参1：真实值  参2：预测值  参3：正例标签
print(f'B模型召回率：{recall_score(y_train, y_pred_B, pos_label = '恶性')}')
print(f'B模型f1：{f1_score(y_train, y_pred_B, pos_label = '恶性')}')
