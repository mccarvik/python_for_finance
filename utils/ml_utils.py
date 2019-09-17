"""
Module for common ML utilities
"""
import pdb
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
mpl.use('Agg')

IMG_PATH = '/home/ec2-user/environment/python_for_finance/research/ml_analysis/png/temp/'
IMG_ROOT = '/home/ec2-user/environment/python_for_finance/research/ml_analysis/png/'

def update_check(list1, list2):
    """
    Check the diff between two lists for a change in weights
    """
    for i, j in zip(list1, list2):
        if i != j:
            return True
    return False


def plot_decision_regions(x_vals, y_vals, classifier, test_break_idx=None, resolution=0.02):
    """
    Will plot a chart using MatPlotLb showing the decision regions of a classifier
    """
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y_vals))])

    # plot the decision surface
    x1_min, x1_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
    x2_min, x2_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # will plot over entire decision surface, not just given points
    z_vals = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z_vals = z_vals.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z_vals, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, clss in enumerate(np.unique(y_vals)):
        edit_cmap = [list(cmap(idx))]
        plt.scatter(x=x_vals[y_vals == clss, 0], y=x_vals[y_vals == clss, 1],
                    alpha=0.8, c=edit_cmap,
                    marker=markers[idx], label=clss)

    # highlight test samples
    if test_break_idx:
        x_test, _ = x_vals[test_break_idx], y_vals[test_break_idx]
        plt.scatter(x_test[:, 0],
                    x_test[:, 1],
                    facecolors='none',
                    edgecolors='black',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')


def standardize(x_train, x_test=None):
    """
    Standardization of the data --> everything based on std's from the mean
    """
    scale = StandardScaler()
    scale.fit(x_train)
    x_train_std = scale.transform(x_train)
    if x_test != None:
        x_test_std = scale.transform(x_test)
        return (x_train_std, x_test_std)
    return x_train_std


# Decision Tree criterion funcs
def gini(prob):
    """
    Decision Tree gini coefficient
    """
    return (prob) * (1 - (prob)) + (1 - prob) * (1 - (1 - prob))


def entropy(prob):
    """
    Decision Tree gini coefficient
    """
    return - prob * np.log2(prob) - (1 - prob) * np.log2((1 - prob))


def error(prob):
    """
    calculate the error of a given probability
    """
    return 1 - np.max([prob, 1 - prob])


def lin_regplot(x_vals, y_vals, model):
    """
    Plot a linear regression
    """
    plt.scatter(x_vals, y_vals, c='lightblue')
    plt.plot(x_vals, model.predict(x_vals), color='red', linewidth=2)
    return None
