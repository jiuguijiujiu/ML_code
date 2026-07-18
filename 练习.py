from sklearn.preprocessing import StandardScaler

x_train = [[90, 2, 10, 40],
           [60, 4, 15, 45],
           [75, 3, 13, 46]]

transfor = StandardScaler()

x_train_new = transfor.fit_transform(x_train)

print(x_train_new)

print(transfor.mean_)
print(transfor.var_)