"""
Module to implement a Perceptron Classifier
"""
import pdb
import numpy as np
from sklearn import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.ml_utils import plot_decision_regions, IMG_ROOT
mpl.use('Agg')


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
    wgts : 1d-array
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
        self.wgts = []
        self.errors_ = []

    def fit(self, x_vals, y_vals):
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
        self.wgts = np.zeros(1 + x_vals.shape[1])
        # loop through the data n_iter amount of times
        for _ in range(self.n_iter):
            # pdb.set_trace()
            errors = 0
            for x_idx, target in zip(x_vals, y_vals):
                # update only occurs if prediction is wrong
                update = self.eta * (target - self.predict(x_idx))
                # update wgts --> (update) * learning rate (eta)  * factor value (x_idx)
                self.wgts[1:] += update * x_idx
                # update intercept by amount prediction was off * learning rate
                self.wgts[0] += update
                # check if weights were updated
                if update != 0:
                    print(str(self.wgts) + "---- epoch: " + str(_))
                # keeps track of how many mistakes per epoch
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, x_vals):
        """
        Calculate net input
        """
        # simply multiples weights by factors --> = w1x1 + w2x2 + ... + w0
        return np.dot(x_vals, self.wgts[1:]) + self.wgts[0]

    def predict(self, x_vals):
        """
        Return class label after unit step
        Can only return 1 0r 0 --> different than Adaline
        """
        # binary classifier
        return np.where(self.net_input(x_vals) >= 0.0, 1, -1)

    def score(self, x_vals, y_vals):
        """
        Returns the accuracy score of the perceptron
        """
        wrong = 0
        for i, j in zip(x_vals, y_vals):
            if j != self.predict(i):
                wrong += 1
        return (len(y_vals) - wrong) / len(y_vals)


if __name__ == '__main__':
    IRIS = datasets.load_iris()
    Y_VALS = IRIS.target[0:100]
    Y_VALS = np.where(Y_VALS == 0, -1, 1)
    X_VALS = IRIS.data[0:100, [0, 2]]

    # Plot the x vals of the data set
    plt.scatter(X_VALS[:50, 0], X_VALS[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X_VALS[50:100, 0], X_VALS[50:100, 1], color='blue', marker='x', label='versicolor')

    PPN = Perceptron(eta=0.1, n_iter=10)
    PPN.fit(X_VALS, Y_VALS)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plot_decision_regions(X_VALS, Y_VALS, classifier=PPN)
    pdb.set_trace()
    plt.savefig(IMG_ROOT + "PML/" + "iris_ch2.png", dpi=300)
    plt.close()

    plt.plot(range(1, len(PPN.errors_) + 1), PPN.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.savefig(IMG_ROOT + "PML/" + "iris2_ch2.png", dpi=300)
    plt.close()
