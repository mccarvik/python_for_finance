"""
Implement an Adaptive Linear Neuron with Stochastic Gradient Descent
Aka "Iterative" or "On-Line" gradient descent as weights are updated on each sample
"""
# import pdb
import numpy as np
from numpy.random import seed
from sklearn import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.ml_utils import plot_decision_regions, IMG_ROOT
mpl.use('Agg')


IMG_PATH = '/home/ubuntu/workspace/finance/app/static/img/ml_imgs/'

class AdalineSGD(object):
    """
    ADAptive LInear NEuron classifier.

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
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        """
        Constructor
        """
        self.eta = eta
        self.n_iter = n_iter
        self.wgts = []
        self.cost_ = []
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, x_vals, y_vals):
        """
        Fit training data.

        Parameters
        ----------
        x_vals : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y_vals : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """
        self._initialize_weights(x_vals.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            # data needs to be presented in random order to prevent cycles
            if self.shuffle:
                x_vals, y_vals = shuffle_func(x_vals, y_vals)
            cost = []
            for ind_x, target in zip(x_vals, y_vals):
                # update weights "on-the-fly" after each sample, not in a batch
                cost.append(self._update_weights(ind_x, target))
            avg_cost = sum(cost)/len(y_vals)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, x_vals, _vals):
        """
        Fit training data without reinitializing the weights
        """
        # Used to continue learning on a model after weights have already been tuned to some extent
        if not self.w_initialized:
            self._initialize_weights(x_vals.shape[1])
        if _vals.ravel().shape[0] > 1:
            for ind_x, target in zip(x_vals, _vals):
                self._update_weights(ind_x, target)
        else:
            self._update_weights(x_vals, _vals)
        return self

    def _initialize_weights(self, x_len):
        """
        Initialize weights to zeros
        """
        self.wgts = np.zeros(1 + x_len)
        self.w_initialized = True

    def _update_weights(self, ind_x, target):
        """
        Apply Adaline learning with one sample to update the weights
        """
        output = self.net_input(ind_x)
        error = (target - output)
        # weights are updated even if prediction is right, based on how large the error is
        self.wgts[1:] += self.eta * ind_x.dot(error)
        self.wgts[0] += self.eta * error
        cost = 0.5 * error**2
        print(str(self.wgts) + "----" + str(ind_x))
        return cost

    def net_input(self, x_vals):
        """
        Calculate net input
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


def shuffle_func(x_vals, y_vals):
    """
    Shuffle training data
    """
    rand = np.random.permutation(len(y_vals))
    return x_vals[rand], y_vals[rand]


if __name__ == "__main__":
    # Get Data
    IRIS = datasets.load_iris()
    Y_VALS = IRIS.target[0:100]
    Y_VALS = np.where(Y_VALS == 0, -1, 1)
    X_VALS = IRIS.data[0:100, [0, 2]]

    ADA = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ADA.fit(X_VALS, Y_VALS)

    plot_decision_regions(X_VALS, Y_VALS, classifier=ADA)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_ROOT + "PML/" + 'adaline_sgd_4.png', dpi=300)
    plt.close()

    plt.plot(range(1, len(ADA.cost_) + 1), ADA.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.tight_layout()
    plt.savefig(IMG_ROOT + "PML/" +'adaline_sgd_5.png', dpi=300)
    plt.close()
