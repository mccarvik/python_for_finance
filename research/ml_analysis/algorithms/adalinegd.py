import sys, pdb
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.ml_utils import update_check

class AdalineGD(object):
    """ADAptive LInear NEuron classifier
    
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
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
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
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        # loop through the data n_iter amount of times
        for i in range(self.n_iter):
            output = self.net_input(X)
            # errors will accumulate even if in the correct direction, ex:
            # y = 1, output=2.1, err=-1.1 even though would have predicted correct class, helps get to exact output, not just correct prediction
            # Different than perceptron
            errors = (y - output)
            temp = tuple(self.w_)
            # Updates weights based on how far off prediciton is, unlike perceptron
            # Updates the weights once for the entire iteration of data unlike perceptron which does it for each sample
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            if update_check(temp, self.w_):
                print(str(self.w_) + "----" + str(i))
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        # simply multiples weights by factors --> = w1x1 + w2x2 + ... + w0
        # Dot product of the WHOLE dataset, not just each entry
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
def main2():
    y,X = get_data()
    
    # standardize features
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    
    ada = AdalineGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)
    
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(PIC_LOC + 'adaline_2.png', dpi=300)
    plt.close()
    
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    
    plt.tight_layout()
    plt.savefig(PIC_LOC + 'adaline_3.png', dpi=300)
    # plt.show()
    plt.close()

if __name__ == "__main__":
    main2()
    