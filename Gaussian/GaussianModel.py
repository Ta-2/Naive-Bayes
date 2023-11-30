import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import matplotlib.pylab as plt

datas = np.loadtxt("iris.csv")
#print(datas)
X = datas[:,0:4]
#print(X)
Y = datas[:,-1].astype("int32")
#print(Y)
G = GaussianNB(priors=None, var_smoothing=1e-09)
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

for s in random_seed:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=s)
    G.fit(X_train, Y_train)
    Y_pred_GNB = G.predict(X_test)
    res = classification_report(Y_test, Y_pred_GNB, output_dict=True)
    setosa.add_data(res['0']["f1-score"], res['0']["precision"], res['0']["recall"])
    versicolor.add_data(res['1']["f1-score"], res['1']["precision"], res['1']["recall"])
    virginica.add_data(res['2']["f1-score"], res['2']["precision"], res['2']["recall"])

print(setosa)
print(versicolor)
print(virginica)

def Normal(x, theta, var):
    return (1/np.sqrt(2*np.pi*var))*np.exp(-(x-theta)**2 / (2*var**2))

print("theta:")
print(G.theta_)
print("var:")
print(G.var_)
print()
theta = G.theta_
var = G.var_

X = []
Y = []
for c_t, c_v in zip(theta, var):
    add_x = []
    add_y = []
    for p_t, p_v in zip(c_t, c_v):
        add_x.append(np.linspace(p_t-1, p_t+1, 200))
        add_y.append([Normal(x, p_t, p_v) for x in add_x[-1]])
    X.append(add_x)
    Y.append(add_y)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
class_num = 3
param_num = 4
for p in range(param_num):
    for c in range(class_num):
        ax[p//2,p%2].plot(X[c][p], Y[c][p])
        if p==0:
            ax[p//2,p%2].set_title("sepal_Length")
        if p==1:
            ax[p//2,p%2].set_title("sepal_Width")
        if p==2:
            ax[p//2,p%2].set_title("Petal_Length")
        if p==3:
            ax[p//2,p%2].set_title("Petal_Width")

plt.show()