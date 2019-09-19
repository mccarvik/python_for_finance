"""
Script to use Neural Network code
"""
import pdb
import os
import gzip
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.ml_utils import IMG_ROOT
from research.ml_analysis.dev_work.neural_net import NeuralNetMLP
mpl.use('Agg')


def load_mnist(path, kind='train'):
    """
    Load MNIST data from `path`
    """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)

    with gzip.open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype=np.uint8).reshape(len(labels), 784).astype(np.float64)
    return images, labels


def get_mnist(images=False):
    """
    get the mnist data
    """
    x_train, y_train = load_mnist('dev_data/', kind='train')
    print('Rows: %d, columns: %d' % (x_train.shape[0], x_train.shape[1]))
    x_test, y_test = load_mnist('dev_data/', kind='t10k')
    print('Rows: %d, columns: %d' % (x_test.shape[0], x_test.shape[1]))

    if images:
        # plot first digit of each class
        _, axx = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
        axx = axx.flatten()
        for i in range(10):
            img = x_train[y_train == i][0].reshape(28, 28)
            axx[i].imshow(img, cmap='Greys', interpolation='nearest')
        axx[0].set_xticks([])
        axx[0].set_yticks([])
        plt.tight_layout()
        plt.savefig(IMG_ROOT + "PML/" + 'mnist_test.png', dpi=300)

        # plot first 25 7's
        _, axx2 = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
        axx2 = axx2.flatten()
        for i in range(25):
            img = x_train[y_train == 7][i].reshape(28, 28)
            axx2[i].imshow(img, cmap='Greys', interpolation='nearest')

        axx2[0].set_xticks([])
        axx2[0].set_yticks([])
        plt.tight_layout()
        plt.savefig(IMG_ROOT + "PML/" + 'mnist_7.png', dpi=300)
    return x_train, y_train, x_test, y_test


def nn_mlp_ex(val_len=1000, plot=False, acc=True):
    """
    Function to run the Neural Network MLP implementation
    """
    x_train, y_train, x_test, y_test = get_mnist()
    if val_len:
        x_train = x_train[:val_len]
        y_train = y_train[:val_len]

    neural_net = NeuralNetMLP(n_output=10, n_features=x_train.shape[1], n_hidden=50, ll2=0.1,
                              ll1=0.0, epochs=10, eta=0.001, alpha=0.001, decrease_const=0.00001,
                              minibatches=50, shuffle=True, random_state=1)
    neural_net.fit(x_train, y_train, print_progress=True)

    if plot:
        pdb.set_trace()
        plt.plot(range(len(neural_net.cost_)), neural_net.cost_)
        plt.ylim([0, 2000])
        plt.ylabel('Cost')
        plt.xlabel('Epochs * 50')
        plt.tight_layout()
        plt.savefig(IMG_ROOT + "PML/" + 'nn_batch_cost.png', dpi=300)
        plt.close()

        # batches = np.array_split(range(len(neural_net.cost_)), 1000)
        # cost_ary = np.array(neural_net.cost_)
        # cost_avgs = [np.mean(cost_ary[i]) for i in batches]
        # plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
        # plt.ylim([0, 2000])
        # plt.ylabel('Cost')
        # plt.xlabel('Epochs')
        # plt.tight_layout()
        # plt.savefig(IMG_ROOT + "PML/" + 'nn_batch_cost2.png', dpi=300)
        # plt.close()

    if acc:
        print()
        y_train_pred = neural_net.predict(x_train)
        acc_train = np.sum(y_train == y_train_pred, axis=0) / x_train.shape[0]
        print('Training accuracy: %.2f%%' % (acc_train * 100))

        y_test_pred = neural_net.predict(x_test)
        acc_test = np.sum(y_test == y_test_pred, axis=0) / x_test.shape[0]
        print('Test accuracy: %.2f%%' % (acc_test * 100))


if __name__ == '__main__':
    # get_mnist(images=True)
    nn_mlp_ex()
