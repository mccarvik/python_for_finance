"""
Implement an Adaptive Linear Neuron with Gradient Descent
aka "Batch Gradient Descent" as the whole batch is evaluated on each epoch
"""
import pdb
import numpy as np
from sklearn import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.ml_utils import plot_decision_regions, IMG_ROOT
mpl.use('Agg')


class AdalineGD(object):
    """
    ADAptive LInear NEuron classifier

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
    def __init__(self, eta=0.01, n_iter=50):
        """
        Constructor
        """
        self.eta = eta
        self.n_iter = n_iter
        self.cost_ = []
        self.wgts = []

    def fit(self, x_vals, y_vals):
        """ Fit training data.

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
            # calculate the output for whole batch based on current weights
            output = self.net_input(x_vals)
            # errors will accumulate even if in the correct direction
            # ex: y = 1, output=2.1, err=-1.1 even though would have predicted correct class
            errors = (y_vals - output)
            # Updates weights based on how far off prediciton is, unlike perceptron
            # Dot product on x vals and error helps deduce how far off the weights are
            self.wgts[1:] += self.eta * x_vals.T.dot(errors)
            self.wgts[0] += self.eta * errors.sum()
            print(str(self.wgts) + "---- epoch: " + str(_))
            # cost --> sum of squered errors (SSE) * 1/2
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, x_vals):
        """
        Calculate net input
        Multiply weights by factors --> = w1x1 + w2x2 + ... + w0
        Dot product of the WHOLE dataset, not just each entry, "Batch GD"
        """
        return np.dot(x_vals, self.wgts[1:]) + self.wgts[0]

    def activation(self, x_vals):
        """
        Compute linear activation
        """
        return self.net_input(x_vals)

    def predict(self, x_vals):
        """
        Return class label after unit step
        """
        return np.where(self.activation(x_vals) >= 0.0, 1, -1)


if __name__ == "__main__":
    # Get Data
    IRIS = datasets.load_iris()
    Y_VALS = IRIS.target[0:100]
    Y_VALS = np.where(Y_VALS == 0, -1, 1)
    X_VALS = IRIS.data[0:100, [0, 2]]

    #  Perform Comparison on diff learning rates
    FIG, AX = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ADA1 = AdalineGD(n_iter=10, eta=0.01).fit(X_VALS, Y_VALS)
    AX[0].plot(range(1, len(ADA1.cost_) + 1), np.log10(ADA1.cost_), marker='o')
    AX[0].set_xlabel('Epochs')
    AX[0].set_ylabel('log(Sum-squared-error)')
    AX[0].set_title('Adaline - Learning rate 0.01')

    ADA2 = AdalineGD(n_iter=10, eta=0.0001).fit(X_VALS, Y_VALS)
    AX[1].plot(range(1, len(ADA2.cost_) + 1), ADA2.cost_, marker='o')
    AX[1].set_xlabel('Epochs')
    AX[1].set_ylabel('Sum-squared-error')
    AX[1].set_title('Adaline - Learning rate 0.0001')
    plt.tight_layout()
    plt.savefig(IMG_ROOT + "PML/" + 'adaline_comp.png', dpi=300)
    plt.close()

    # standardize features
    X_STD = np.copy(X_VALS)
    X_STD[:, 0] = (X_VALS[:, 0] - X_VALS[:, 0].mean()) / X_VALS[:, 0].std()
    X_STD[:, 1] = (X_VALS[:, 1] - X_VALS[:, 1].mean()) / X_VALS[:, 1].std()

    # Implement AdalineGD on Standardized data
    ADA = AdalineGD(n_iter=15, eta=0.01)
    ADA.fit(X_STD, Y_VALS)
    plot_decision_regions(X_STD, Y_VALS, classifier=ADA)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_ROOT + "PML/" + 'adaline_2.png', dpi=300)
    plt.close()

    plt.plot(range(1, len(ADA.cost_) + 1), ADA.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.tight_layout()
    plt.savefig(IMG_ROOT + "PML/" + 'adaline_3.png', dpi=300)
    plt.close()
    