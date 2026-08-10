import numpy as np                  # 数学计算包
import pandas as pd                 # 数据处理包
import matplotlib.pyplot as plt     # 画图包
import jieba                        # 分词包
from sklearn.feature_extraction.text import CountVectorizer     # 词频统计包，把评论内容 转成 词频矩阵。
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB                   # 朴素贝叶斯对象

# 1. 加载数据
df = pd.read_csv('./data/书籍评价.csv', encoding='gbk')
# df.info()

# 2. 数据预处理
# 2.1 添加labels列，充当：标签列，好评 → 1，差评 → 0
df['labels'] = np.where(df['评价'] == '好评', 1, 0)
# df.info()
# print(df.head())
# 提取标签列
y = df['labels']

# 2.2 演示jieba分词
# print(jieba.lcut('好好学习，天天向上！我爱你你爱我，蜜雪冰城甜蜜蜜！小明骑车，一把把把把住了！'))

# 2.3 对用户的评论信息，做切词。
# 数据格式：[[第1条评论切词1，切词2，切词3...]，[第2条评论切词1，切词2，切词3...]，...]
# comment_list1 = [jieba.lcut(line) for line in df['内容']]
# 演示字符串的 join()函数用法.
# my_list = ['aa', 'bb', 'cc']
# print(','.join(my_list))
comment_list = [','.join(jieba.lcut(line)) for line in df['内容']]
# print(comment_list)

# 2.4 加载 停用词列表，即：里边记录的词，不需要参与模型训练、预测，要被删除的词，例如：的，啊，哈，从，都...
with open('./data/stopwords.txt', 'r', encoding = 'utf-8') as src_f:
    # 2.4.1 一次读取所有的行
    stopwords_list = src_f.readlines()
    # 2.4.2 删除最后的 '\n'
    stopwords_list = [x.strip() for x in stopwords_list]
    # 2.4.3 对停用词列表去重
    stopwords_list = list(set(stopwords_list))
    # print(stopwords_list)

# 2.5 创建向量化对象，从 评论切词列表(comment_list) 中 删除 停用词，并且统计词频(单词矩阵)。
transfor = CountVectorizer(stop_words = stopwords_list)

# 2.6 统计词频矩阵，先训练，后转换，在转数组。
# transfor.fit(comment_list)
# x的格式：[[第1条评论的切词分布，有就是1，没有就是0]，[第2条评论的切词分布，有就是1，没有就是0]，...]
# x = transfor.transform(comment_list).toarray()
x = transfor.fit_transform(comment_list).toarray()
# print(x)

# 2.7 看一下，我们13条评论，切词，且删除 停用词后，一共剩下多少个词了。
# print(transfor.get_feature_names_out())
# print(len(transfor.get_feature_names_out()))            # 37个词，即：13条评论，切词，且删除 停用词后，一共剩下多少个词了。

# 2.8 切分数据集
x_train = x[:10]
y_train = y[:10]

x_test = x[10:]
y_test = y[10:]

# 3. 特征工程,省略

# 4. 模型训练
estimator = MultinomialNB()     # 创建朴素贝叶斯模型对象
estimator.fit(x_train, y_train)

# 5. 模型预测
y_pred = estimator.predict(x_test)
print(f'预测结果:{y_pred}')

# 6. 模型评估
print(f'准确率:{accuracy_score(y_test, y_pred)}')