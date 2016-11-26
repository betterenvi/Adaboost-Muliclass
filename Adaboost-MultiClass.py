# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from pandas import Series, DataFrame
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import string

train_fn = './Downloads/letter-recognition.data'
data_col = ['letter','x-box','y-box','width','high','onpix','x-bar','y-bar',
            'x2bar','y2bar','xybar','x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx']
X = pd.read_csv(train_fn, sep=',', header=None, names=data_col,
                     skiprows=None, na_values='?', keep_default_na=False, engine='python')


y = X['letter']
y.head()

cols = list(X)
cols.pop(cols.index('letter'))
X = X[cols]
X.head()
#get_ipython().magic(u'pinfo train_test_split')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_train.head()
X_train.shape, X_test.shape

def models_N_weights(X, y, M, k, max_depth):
    model = []
    model_weights = []
    training_errors = []

    N = X.shape[0]
    w = np.ones(N) / N

    for m in range(M):
        h = DecisionTreeClassifier(max_depth=max_depth)
        h.fit(X, y, sample_weight=w)
        pred = h.predict(X)

        eps = w.dot(pred != y)
        alpha = (np.log((1 - eps)*(k - 1)) - np.log(eps)) / 2
        w_new = np.where(y == pred, w * np.exp(-alpha), w * np.exp(alpha))
        w = w_new / w_new.sum()

        model.append(h)
        model_weights.append(alpha)

    return [model, model_weights]



def predict_joined_models(X, model, model_weights, frame, m):
    pred = model[m].predict(X)
    for i, idx in enumerate(frame.index):
        t = frame.get_value(idx, pred[i])
        frame.set_value(idx, pred[i], t + model_weights[m])
    #获取每行最大数据的列名
    return frame.idxmax(axis=1)



def error_func(y, y_hat):
    correct_pred = (np.array(y_hat) == np.array(y))
    Err = 1 - float(sum(correct_pred))/len(correct_pred)
    return Err



models = dict()
train_errs = dict()
test_errs = dict()
for max_depth in range(1, 6):
    M = 100
    k = 26
    M_list = []
    train_err_list = []
    test_err_list = []
    N1= X_train.shape[0]
    frame1 = DataFrame(np.zeros([N1,26]),columns=list(string.uppercase))
    N2= X_test.shape[0]
    frame2 = DataFrame(np.zeros([N2,26]),columns=list(string.uppercase))

    model_fit = models_N_weights(X_train, y_train, M, k, max_depth)
    for m in range(M):
        y_hat = predict_joined_models(X_train, model_fit[0], model_fit[1], frame1, m)
        err = error_func(y_train, y_hat)
        train_err_list.append(err)

        y_hat = predict_joined_models(X_test, model_fit[0], model_fit[1], frame2, m)
        err = error_func(y_test, y_hat)
        test_err_list.append(err)
        M_list.append(m)

    models[max_depth] = M_list
    train_errs[max_depth] = train_err_list
    test_errs[max_depth] = test_err_list



import matplotlib.cm as cm
colors = iter(cm.rainbow(np.linspace(0, 1, len(models) * 2)))
for md in models.keys():
    M_list = models[md]
    train_err_list = train_errs[md]
    test_err_list = test_errs[md]
    plt.plot(M_list, test_err_list, c=next(colors), linestyle='-', label='test, max_depth=%d' % md)
    plt.plot(M_list, train_err_list, c=next(colors), linestyle='--', label='train, max_depth=%d' % md)

plt.xlabel('Number of weak learners')
plt.ylabel('Error')
plt.title('Error and number of weak learners')
plt.legend()
plt.show()
