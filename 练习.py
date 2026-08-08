import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('./data/wine0501.csv')
# data.info()
# print(data['Class label'].unique())

data = data[data['Class label'] != 3]
# print(data.head())
# print(data['Class label'].unique())

x = data[['Alcohol', 'Hue']]        # 酒精与色泽
y = data['Class label']

ls = LabelEncoder()
y = ls.fit_transform(y)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

estimator1 = DecisionTreeClassifier(max_depth = 3)
estimator1.fit(x_train, y_train)
y_pred1 = estimator1.predict(x_test)
print(y_pred1)
print(accuracy_score(y_test, y_pred1))
print('-' * 50)

estimator2 = AdaBoostClassifier(estimator1, n_estimators = 100, learning_rate = 0.1)
estimator2.fit(x_train, y_train)
y_pred2 = estimator2.predict(x_test)
print(y_pred2)
print(accuracy_score(y_test, y_pred2))
