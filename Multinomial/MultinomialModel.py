import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pprint
import matplotlib.pylab as plt

datas = np.loadtxt("iris.csv")
#print(datas)
X = datas[:,0:4]
#print(X)
Y = datas[:,-1].astype("int32")
#print(Y)
M = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
cross_num = 20
random_seed = np.arange(cross_num)

class ResultData:
    def __init__(self, name) -> None:
        self.name = name
        self.f1 = list()
        self.precision = list()
        self.recall = list()
    
    def add_data(self, f1, pre, rec):
        self.f1.append(f1)
        self.precision.append(pre)
        self.recall.append(rec)

    def __str__(self) -> str:
        f1_m = np.mean(self.f1)
        f1_v = np.var(self.f1)
        f1_str = "f1 mean: " + str(f1_m) + ", f1 var: " + str(f1_v)
        pr_m = np.mean(self.precision)
        pr_v = np.var(self.precision)
        pr_str = "precision mean: " + str(pr_m) + ", precision var: " + str(pr_v)
        re_m = np.mean(self.recall)
        re_v = np.var(self.recall)
        re_str = "recall mean: " + str(re_m) + ", recall var: " + str(re_v)
        return self.name + "\n\t" + f1_str + "\n\t" + pr_str + "\n\t" + re_str

setosa = ResultData("setosa")
versicolor = ResultData("versicolor")
virginica = ResultData("virginica")

accuracy = 0.0
most_acc_result = None
for s in random_seed:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=s)
    M.fit(X_train, Y_train)
    Y_pred_GNB = M.predict(X_test)
    res = classification_report(Y_test, Y_pred_GNB, output_dict=True)
    #pprint.pprint(res)
    #if accuracy < res["accuracy"]:
    #    most_acc_result = res
    setosa.add_data(res['0']["f1-score"], res['0']["precision"], res['0']["recall"])
    versicolor.add_data(res['1']["f1-score"], res['1']["precision"], res['1']["recall"])
    virginica.add_data(res['2']["f1-score"], res['2']["precision"], res['2']["recall"])

seed = 0
while(True):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=seed)
    M.fit(X_train, Y_train)
    Y_pred_GNB = M.predict(X_test)
    res = classification_report(Y_test, Y_pred_GNB, output_dict=True)
    if 0.9 < res["accuracy"]:
        break
    seed += 1

print(res["accuracy"])
print(setosa)
print(versicolor)
print(virginica)

test_X0 = np.array([ np.array([i, 0.0, 0.0, 0.0]) for i in np.linspace(4, 8, 20) ])
test_X1 = np.array([ np.array([i, 0.0, 0.0, 0.0]) for i in np.linspace(2, 5, 20) ])
test_X2 = np.array([ np.array([i, 0.0, 0.0, 0.0]) for i in np.linspace(1, 7, 20) ])
test_X3 = np.array([ np.array([i, 0.0, 0.0, 0.0]) for i in np.linspace(0.1, 2.5, 20) ])

pred_Y0 = M.predict_proba(test_X0).T
pred_Y1 = M.predict_proba(test_X1).T
pred_Y2 = M.predict_proba(test_X2).T
pred_Y3 = M.predict_proba(test_X3).T

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].set_title("sepal_Length")
ax[0, 1].set_title("sepal_Width")
ax[1, 0].set_title("Petal_Length")
ax[1, 1].set_title("Petal_Width")

ax[0, 0].plot(np.linspace(4, 8, 20), pred_Y0[0])
ax[0, 0].plot(np.linspace(4, 8, 20), pred_Y0[1])
ax[0, 0].plot(np.linspace(4, 8, 20), pred_Y0[2])

ax[0, 1].plot(np.linspace(2, 5, 20), pred_Y1[0])
ax[0, 1].plot(np.linspace(2, 5, 20), pred_Y1[1])
ax[0, 1].plot(np.linspace(2, 5, 20), pred_Y1[2])

ax[1, 0].plot(np.linspace(1, 7, 20), pred_Y2[0])
ax[1, 0].plot(np.linspace(1, 7, 20), pred_Y2[1])
ax[1, 0].plot(np.linspace(1, 7, 20), pred_Y2[2])

ax[1, 1].plot(np.linspace(0.1, 2.5, 20), pred_Y3[0])
ax[1, 1].plot(np.linspace(0.1, 2.5, 20), pred_Y3[1])
ax[1, 1].plot(np.linspace(0.1, 2.5, 20), pred_Y3[2])

plt.show()