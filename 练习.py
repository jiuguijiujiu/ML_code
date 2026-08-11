import numpy as np                  # 数学计算包
import pandas as pd                 # 数据处理包
import matplotlib.pyplot as plt     # 画图包
import jieba                        # 分词包
from sklearn.feature_extraction.text import CountVectorizer     # 词频统计包，把评论内容 转成 词频矩阵。
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB                   # 朴素贝叶斯对象

data = pd.read_csv('./data/书籍评价.csv', encoding='gbk')
# print(data.head())
# data.info()

data['table'] = np.where(data['评价'] == '好评', 1, 0)
# print(data.head())
y = data['table']

# comment_list = [jieba.lcut(line) for line in data['内容']]
# print(comment_list)

comment_list = [','.join(jieba.lcut(line)) for line in data['内容']]
# print(comment_list)

with open('./data/stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.readlines()
    # print(stopwords)
    stopwords = [x.strip() for x in stopwords]
    # print(stopwords)
    stopwords = list(set(stopwords))
    # print(stopwords)

transfor = CountVectorizer(stop_words=stopwords)
x = transfor.fit_transform(comment_list).toarray()
# print(x)

x_train = x[:10]
y_train = y[:10]

x_test = x[10:]
y_test = y[10:]

estimator = MultinomialNB()
estimator.fit(x_train, y_train)

y_pred = estimator.predict(x_test)
print(y_pred)

print(accuracy_score(y_test, y_pred))