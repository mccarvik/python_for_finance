"""
Neural Network - Multilayer Perceptron
"""
import pdb
import sys
import numpy as np
from scipy.special import expit



class NeuralNetMLP(object):
    """
    Feedforward neural network / Multi-layer perceptron classifier

    Parameters
    ------------
    n_output : int
        Number of output units, should be equal to the
        number of unique class labels.
    n_features : int
        Number of features (dimensions) in the target dataset.
        Should be equal to the number of columns in the X array.
    n_hidden : int (default: 30)
        Number of hidden units.
    l1 : float (default: 0.0)
        Lambda value for L1-regularization.
        No regularization if l1=0.0 (default)
    l2 : float (default: 0.0)
        Lambda value for L2-regularization.
        No regularization if l2=0.0 (default)
    epochs : int (default: 500)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    alpha : float (default: 0.0)
        Momentum constant. Factor multiplied with the
        gradient of the previous epoch t-1 to improve
        learning speed
        w(t) := w(t) - (grad(t) + alpha*grad(t-1))
    decrease_const : float (default: 0.0)
        Decrease constant. Shrinks the learning rate
        after each epoch via eta / (1 + epoch*decrease_const)
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatches : int (default: 1)
        Divides training data into k minibatches for efficiency.
        Normal gradient descent learning if k=1 (default).
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.
    Attributes
    -----------
    cost_ : list
      Sum of squared errors after each epoch.
    """
    def __init__(self, n_output, n_features, n_hidden=30,
                 ll1=0.0, ll2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0, shuffle=True,
                 minibatches=1, random_state=None):
        """
        Constructor
        """
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.wgt1, self.wgt2 = self._initialize_weights()
        self.ll1 = ll1
        self.ll2 = ll2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches
        self.cost_ = []

    def _initialize_weights(self):
        """
        Initialize weights with small random numbers
        """
        # random wgts from input layer to hidden layer
        wgt1 = np.random.uniform(-1.0, 1.0,
                                 size=self.n_hidden*(self.n_features + 1))
        wgt1 = wgt1.reshape(self.n_hidden, self.n_features + 1)
        # Random wgts from hidden layer to output layer
        wgt2 = np.random.uniform(-1.0, 1.0,
                                 size=self.n_output*(self.n_hidden + 1))
        wgt2 = wgt2.reshape(self.n_output, self.n_hidden + 1)
        return wgt1, wgt2

    def _feedforward(self, X, wgt1, wgt2):
        """
        Compute feedforward step

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        wgt1 : array, shape = [n_hidden_units, n_features]
            Weight matrix for input layer -> hidden layer.
        wgt2 : array, shape = [n_output_units, n_hidden_units]
            Weight matrix for hidden layer -> output layer.
        Returns
        ----------
        a1 : array, shape = [n_samples, n_features+1]
            Input values with bias unit.
        z2 : array, shape = [n_hidden, n_samples]
            Net input of hidden layer.
        a2 : array, shape = [n_hidden+1, n_samples]
            Activation of hidden layer.
        z3 : array, shape = [n_output_units, n_samples]
            Net input of output layer.
        a3 : array, shape = [n_output_units, n_samples]
            Activation of output layer.
        """
        pdb.set_trace()
        a1 = add_bias_unit(X, how='column')
        z2 = wgt1.dot(a1.T)
        a2 = sigmoid(z2)
        a2 = add_bias_unit(a2, how='row')
        z3 = wgt2.dot(a2)
        a3 = sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, wgt1, wgt2):
        """Compute L2-regularization cost"""
        # pdb.set_trace()
        return (lambda_/2.0) * (np.sum(wgt1[:, 1:] ** 2) +
                                np.sum(wgt2[:, 1:] ** 2))

    def _L1_reg(self, lambda_, wgt1, wgt2):
        """Compute L1-regularization cost"""
        # pdb.set_trace()
        return (lambda_/2.0) * (np.abs(wgt1[:, 1:]).sum() +
                                np.abs(wgt2[:, 1:]).sum())

    def _get_cost(self, y_enc, output, wgt1, wgt2):
        """Compute cost function.
        Parameters
        ----------
        y_enc : array, shape = (n_labels, n_samples)
            one-hot encoded class labels.
        output : array, shape = [n_output_units, n_samples]
            Activation of the output layer (feedforward)
        wgt1 : array, shape = [n_hidden_units, n_features]
            Weight matrix for input layer -> hidden layer.
        wgt2 : array, shape = [n_output_units, n_hidden_units]
            Weight matrix for hidden layer -> output layer.
        Returns
        ---------
        cost : float
            Regularized cost.
        """
        # pdb.set_trace()
        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.ll1, wgt1, wgt2)
        L2_term = self._L2_reg(self.ll2, wgt1, wgt2)
        cost = cost + L1_term + L2_term
        return cost

    def _get_gradient(self, aa1, aa2, aa3, zz2, y_enc, wgt1, wgt2):
        """ Compute gradient step using backpropagation.
        Parameters
        ------------
        aa1 : array, shape = [n_samples, n_features+1]
            Input values with bias unit.
        aa2 : array, shape = [n_hidden+1, n_samples]
            Activation of hidden layer.
        aa3 : array, shape = [n_output_units, n_samples]
            Activation of output layer.
        zz2 : array, shape = [n_hidden, n_samples]
            Net input of hidden layer.
        y_enc : array, shape = (n_labels, n_samples)
            one-hot encoded class labels.
        wgt1 : array, shape = [n_hidden_units, n_features]
            Weight matrix for input layer -> hidden layer.
        wgt2 : array, shape = [n_output_units, n_hidden_units]
            Weight matrix for hidden layer -> output layer.
        Returns
        ---------
        grad1 : array, shape = [n_hidden_units, n_features]
            Gradient of the weight matrix wgt1.
        grad2 : array, shape = [n_output_units, n_hidden_units]
            Gradient of the weight matrix wgt2.
        """
        # pdb.set_trace()
        # backpropagation
        sigma3 = aa3 - y_enc
        zz2 = add_bias_unit(zz2, how='row')
        sigma2 = wgt2.T.dot(sigma3) * sigmoid_gradient(zz2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(aa1)
        grad2 = sigma3.dot(aa2.T)

        # regularize
        grad1[:, 1:] += self.ll2 * wgt1[:, 1:]
        grad1[:, 1:] += self.ll1 * np.sign(wgt1[:, 1:])
        grad2[:, 1:] += self.ll2 * wgt2[:, 1:]
        grad2[:, 1:] += self.ll1 * np.sign(wgt2[:, 1:])
        return grad1, grad2

    def predict(self, X):
        """
        Predict class labels

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        Returns:
        ----------
        y_pred : array, shape = [n_samples]
            Predicted class labels.
        """
        # pdb.set_trace()
        if len(X.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                 'Use X[:,None] for 1-feature classification,'
                                 '\nor X[[i]] for 1-sample classification')

        a1, z2, a2, z3, a3 = self._feedforward(X, self.wgt1, self.wgt2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, x_vals, y_vals, print_progress=False):
        """
        Learn weights from training data

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        y : array, shape = [n_samples]
            Target class labels.
        print_progress : bool (default: False)
            Prints progress as the number of epochs
            to stderr.
        Returns:
        ----------
        self
        """
        x_data, y_data = x_vals.copy(), y_vals.copy()
        y_enc = encode_labels(y_vals, self.n_output)

        delta_w1_prev = np.zeros(self.wgt1.shape)
        delta_w2_prev = np.zeros(self.wgt2.shape)

        for i in range(self.epochs):
            # adaptive learning rate - rate will decrease over time
            self.eta /= (1 + self.decrease_const*i)

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()

            # Shuffle the data
            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                x_data, y_enc = x_data[idx], y_enc[:, idx]

            pdb.set_trace()
            # split into minibatches
            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                # feedforward
                aa1, zz2, aa2, zz3, aa3 = self._feedforward(x_data[idx], self.wgt1,
                                                            self.wgt2)
                cost = self._get_cost(y_enc=y_enc[:, idx], output=aa3,
                                      wgt1=self.wgt1, wgt2=self.wgt2)
                self.cost_.append(cost)

                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(aa1=aa1, aa2=aa2, aa3=aa3, zz2=zz2,
                                                  y_enc=y_enc[:, idx], wgt1=self.wgt1,
                                                  wgt2=self.wgt2)

                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.wgt1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.wgt2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
        return self


def sigmoid(z_val):
    """
    Compute logistic function (sigmoid)
    Uses scipy.special.expit to avoid overflow error for very small input values z
    """
    pdb.set_trace()
    # return 1.0 / (1.0 + np.exp(-z))
    return expit(z_val)


def sigmoid_gradient(z_val):
    """
    Compute gradient of the logistic function
    """
    pdb.set_trace()
    sgm = sigmoid(z_val)
    return sgm * (1.0 - sgm)


def encode_labels(y_vals, k_out):
    """
    Encode labels into one-hot representation
    Basically binary flags for which output each sample is
    ex: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] is 2 for this sample

    Parameters
    ------------
    y : array, shape = [n_samples]
        Target values.
    Returns
    -----------
    onehot : array, shape = (n_labels, n_samples)
    """
    onehot = np.zeros((k_out, y_vals.shape[0]))
    for idx, val in enumerate(y_vals):
        onehot[val, idx] = 1.0
    return onehot


def add_bias_unit(x_vals, how='column'):
    """
    Add bias unit (column or row of 1s) to array at index 0
    """
    pdb.set_trace()
    if how == 'column':
        x_new = np.ones((x_vals.shape[0], x_vals.shape[1] + 1))
        x_new[:, 1:] = x_vals
    elif how == 'row':
        x_new = np.ones((x_vals.shape[0] + 1, x_vals.shape[1]))
        x_new[1:, :] = x_vals
    else:
        raise AttributeError('`how` must be `column` or `row`')
    return x_new
