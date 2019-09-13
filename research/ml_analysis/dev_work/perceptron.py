"""
"""
import pdb
import numpy as np
import pandas as pd
from sklearn import datasets
from utils.ml_utils import update_check


class Perceptron(object):
    """
    Perceptron classifier

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.01, n_iter=10):
        """
        Constructor
        """
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = []
        self.errors_ = []

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        # set weights to zero
        self.w_ = np.zeros(1 + X.shape[1])
        # loop through the data n_iter amount of times
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                temp = tuple(self.w_)
                # self.predict can only be 1 0r 0 --> different than Adaline
                # update only occurs if prediction is wrong,
                # recalibrates weights accordingly
                update = self.eta * (target - self.predict(xi))
                # update each coefficient weight by amount prediction was off
                # (update) * learning rate (eta)  * factor value (xi)
                self.w_[1:] += update * xi
                # update intercept by amount prediction was off * learning rate
                self.w_[0] += update
                if update_check(temp, self.w_):
                    print(str(self.w_) + "----" + str(_))
                # keeps track of how many mistakes per epoch
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate net input
        """
        # simply multiples weights by factors --> = w1x1 + w2x2 + ... + w0
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Return class label after unit step
        """
        # binary classifier
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def score(self, X, y):
        """
        Returns the accuracy score of the perceptron
        """
        wrong = 0
        for i, j in zip(X, y):
            if j != self.predict(i):
                wrong += 1
        return (len(y) - wrong) / len(y)


if __name__ == '__main__':
    IRIS = datasets.load_iris()
    
    Y_VALS = IRIS.target[0:100]
    Y_VALS = np.where(Y_VALS == 0, -1, 1)
    pdb.set_trace()
    X = df.iloc[0:100, [0, 2]].values
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X,y)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plot_decision_regions(X, y, classifier=ppn)
    plt.savefig(PIC_LOC + "iris_ch2.png")
    plt.close()
    
    plt.plot(range(1,len(ppn.errors_) + 1), ppn.errors_,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.savefig(PIC_LOC + "iris2_ch2.png")
    plt.close()
