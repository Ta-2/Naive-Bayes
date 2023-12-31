import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report

datas = np.loadtxt("iris.csv")
print(datas)
X = datas[:,0:4]
print(X)
Y = datas[:,-1].astype("int32")
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=3)

B = BernoulliNB(priors=None, var_smoothing=1e-09)

B.fit(X_train, Y_train)
B.get_params(deep=True)

Y_pred_GNB = B.predict(X_test)
print(Y_pred_GNB)
print(classification_report(Y_test, Y_pred_GNB))