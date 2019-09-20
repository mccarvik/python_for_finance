"""
Neural Network - Multilayer Perceptron
"""
import pdb
import sys
import numpy as np
from scipy.special import expit

####### NOTES ######
# net-input = values of the layer
# wgts = values of the connections between layers
####################

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
                 minibatches=1, random_state=None, grad_check=False):
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
        self.grad_check = grad_check

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

    def _feed_forward(self, x_vals, wgts=None):
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
        # get the values from input layer (sample data)
        aa1 = add_bias_unit(x_vals, how='column')
        # calc "net input" from dot product with current weights
        if not wgts:
            zz2 = self.wgt1.dot(aa1.T)
        else:
            zz2 = wgts[0].dot(aa1.T)
        # Pass net input to activation function to get values for hidden layer
        aa2 = sigmoid(zz2)
        # add y-intercept/bias unit for values going to output layer
        aa2 = add_bias_unit(aa2, how='row')
        # calc "net_input" going to output layer with dot product of weights
        if not wgts:
            zz3 = self.wgt2.dot(aa2)
        else:
            zz3 = wgts[1].dot(aa2)
        # Pass to activation function
        aa3 = sigmoid(zz3)
        # return everything, including output values
        return aa1, zz2, aa2, zz3, aa3

    def _l2_reg(self, wgt1, wgt2):
        """
        Compute L2-regularization cost
        Takes the sum of the squares of all weights, multiplies by lambda and
        adds it to cost function
        This punishes algo from having weights to strong, preventing overfitting
        """
        return (self.ll2 / 2.0) * (np.sum(wgt1[:, 1:] ** 2) + np.sum(wgt2[:, 1:] ** 2))

    def _l1_reg(self, wgt1, wgt2):
        """
        Compute L1-regularization cost
        Takes the sum of the absolute values of all weights, multiplies by lambda and
        adds it to cost function
        This punishes algo from having weights to strong, preventing overfitting
        """
        return (self.ll1 / 2.0) * (np.abs(wgt1[:, 1:]).sum() + np.abs(wgt2[:, 1:]).sum())

    def _get_cost(self, y_enc, output, wgt1, wgt2):
        """
        Compute cost function.

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
        # Cost function is the logistic cost function
        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2)
        # Add regularization terms to punish stronger weights and avoid over fitting
        l1_term = self._l1_reg(wgt1, wgt2)
        l2_term = self._l2_reg(wgt1, wgt2)
        cost = cost + l1_term + l2_term
        return cost

    def _get_gradient(self, aa1, aa2, aa3, zz2, y_enc, wgt1, wgt2):
        """
        Compute gradient step using backpropagation.

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
        # backpropagation
        # First calculate the error vector
        sigma3 = aa3 - y_enc
        # add y-intercept feature
        zz2 = add_bias_unit(zz2, how='row')
        # calculate the error term of the hidden layer
        # (dot product of wgts * error vector) * gradient of net input of hidden layer
        sigma2 = wgt2.T.dot(sigma3) * sigmoid_gradient(zz2)
        # remove bias unit
        sigma2 = sigma2[1:, :]
        # dot product of error term of hidden layer with input values
        grad1 = sigma2.dot(aa1)
        # dot product of error output vector with activation of hidden layer
        grad2 = sigma3.dot(aa2.T)

        # add regularization values for each gradient to be used for updating weights
        grad1[:, 1:] += self.ll2 * wgt1[:, 1:]
        grad1[:, 1:] += self.ll1 * np.sign(wgt1[:, 1:])
        grad2[:, 1:] += self.ll2 * wgt2[:, 1:]
        grad2[:, 1:] += self.ll1 * np.sign(wgt2[:, 1:])
        return grad1, grad2

    def predict(self, x_vals):
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
        if len(x_vals.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                 'Use X[:,None] for 1-feature classification,'
                                 '\nor X[[i]] for 1-sample classification')

        _, _, _, zz3, _ = self._feed_forward(x_vals)
        y_pred = np.argmax(zz3, axis=0)
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

            # split into minibatches
            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                # Feed forward
                aa1, zz2, aa2, _, aa3 = self._feed_forward(x_data[idx])
                # Calculate the cost of current weights
                cost = self._get_cost(y_enc=y_enc[:, idx], output=aa3,
                                      wgt1=self.wgt1, wgt2=self.wgt2)
                self.cost_.append(cost)

                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(aa1=aa1, aa2=aa2, aa3=aa3, zz2=zz2,
                                                  y_enc=y_enc[:, idx], wgt1=self.wgt1,
                                                  wgt2=self.wgt2)

                if self.grad_check:
                    # start gradient checking
                    grad_diff = self._gradient_checking(x_vals=x_data[idx], y_enc=y_enc[:, idx],
                                                        epsilon=1e-5, grad1=grad1, grad2=grad2)
                    if grad_diff <= 1e-7:
                        print('Ok: %s' % grad_diff)
                    elif grad_diff <= 1e-4:
                        print('Warning: %s' % grad_diff)
                    else:
                        print('PROBLEM: %s' % grad_diff)

                # Apply learning rate
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                # update weights
                # alpha - constant applied to improve learning speed
                self.wgt1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.wgt2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
        return self

    def _gradient_checking(self, x_vals, y_enc, epsilon, grad1, grad2):
        """
        Apply gradient checking (for debugging only)

        Returns
        ---------
        relative_error : float
          Relative error between the numerically
          approximated gradients and the backpropagated gradients.

        """
        num_grad1 = np.zeros(np.shape(self.wgt1))
        epsilon_ary1 = np.zeros(np.shape(self.wgt1))
        for i in range(self.wgt1.shape[0]):
            for j in range(self.wgt1.shape[1]):
                epsilon_ary1[i, j] = epsilon
                _, _, _, _, aa3 = self._feed_forward(x_vals, [self.wgt1 - epsilon_ary1, self.wgt2])
                cost1 = self._get_cost(y_enc, aa3, self.wgt1 - epsilon_ary1, self.wgt2)
                _, _, _, _, aa3 = self._feed_forward(x_vals, [self.wgt1 + epsilon_ary1, self.wgt2])
                cost2 = self._get_cost(y_enc, aa3, self.wgt1 + epsilon_ary1, self.wgt2)
                num_grad1[i, j] = (cost2 - cost1) / (2.0 * epsilon)
                epsilon_ary1[i, j] = 0

        num_grad2 = np.zeros(np.shape(self.wgt2))
        epsilon_ary2 = np.zeros(np.shape(self.wgt2))
        for i in range(self.wgt2.shape[0]):
            for j in range(self.wgt2.shape[1]):
                epsilon_ary2[i, j] = epsilon
                _, _, _, _, aa3 = self._feed_forward(x_vals, [self.wgt1, self.wgt2 - epsilon_ary2])
                cost1 = self._get_cost(y_enc, aa3, self.wgt1, self.wgt2 - epsilon_ary2)
                _, _, _, _, aa3 = self._feed_forward(x_vals, [self.wgt1, self.wgt2 + epsilon_ary2])
                cost2 = self._get_cost(y_enc, aa3, self.wgt1, self.wgt2 + epsilon_ary2)
                num_grad2[i, j] = (cost2 - cost1) / (2.0 * epsilon)
                epsilon_ary2[i, j] = 0

        num_grad = np.hstack((num_grad1.flatten(), num_grad2.flatten()))
        grad = np.hstack((grad1.flatten(), grad2.flatten()))
        norm1 = np.linalg.norm(num_grad - grad)
        norm2 = np.linalg.norm(num_grad)
        norm3 = np.linalg.norm(grad)
        relative_error = norm1 / (norm2 + norm3)
        return relative_error

def sigmoid(z_val):
    """
    Compute logistic function (sigmoid)
    This is our activation function
    Uses scipy.special.expit to avoid overflow error for very small input values z
    """
    # return 1.0 / (1.0 + np.exp(-z))
    return expit(z_val)


def sigmoid_gradient(z_val):
    """
    Compute gradient of the logistic function
    """
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
    Functions as the y-intercept value for the array of inputs for a given sample
    """
    if how == 'column':
        x_new = np.ones((x_vals.shape[0], x_vals.shape[1] + 1))
        x_new[:, 1:] = x_vals
    elif how == 'row':
        x_new = np.ones((x_vals.shape[0] + 1, x_vals.shape[1]))
        x_new[1:, :] = x_vals
    else:
        raise AttributeError('`how` must be `column` or `row`')
    return x_new
