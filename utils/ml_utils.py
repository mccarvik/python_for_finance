import sys, pdb
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler

IMG_PATH = '/home/ubuntu/workspace/ml_dev_work/static/img/temp/'
IMG_ROOT = '/home/ubuntu/workspace/ml_dev_work/static/img/'

def update_check(list1, list2):
    for i,j in zip(list1, list2):
        if i != j:
            return True
    return False

    
def plot_decision_regions(X, y, classifier, test_break_idx=None, resolution=0.02):
   # setup marker generator and color map
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(y))])

   # plot the decision surface
   x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())

   for idx, cl in enumerate(np.unique(y)):
       plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8, c=cmap(idx),
                   marker=markers[idx], label=cl)

   # highlight test samples
   if test_break_idx:
       X_test, y_test = X[test_break_idx:], y[test_break_idx:]
       plt.scatter(X_test[:, 0],
                   X_test[:, 1],
                   c='',
                   alpha=1.0,
                   linewidths=1,
                   marker='o',
                   s=55, label='test set')


def standardize(X_train, X_test=None):
    # Standardization of the data --> everything based on std's from the mean
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    if X_test != None:
        X_test_std = sc.transform(X_test)
        return (X_train_std, X_test_std)
    else:
        return X_train_std


# Decision Tree criterion funcs
def gini(p):
    return (p)*(1 - (p)) + (1-p)*(1 - (1-p))

  
def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

  
def error(p):
    return 1 - np.max([p, 1 - p])


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='lightblue')
    plt.plot(X, model.predict(X), color='red', linewidth=2)
    return None