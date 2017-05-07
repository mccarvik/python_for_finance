import matplotlib as mpl
mpl.use('Agg')
import pdb
import numpy as np
import matplotlib.pyplot as plt

PATH = '/home/ubuntu/workspace/python_for_finance/png/ch5/'

def one_dims():
    np.random.seed(1000)
    y = np.random.standard_normal(20)
    x = range(len(y))
    plt.plot(x, y)
    plt.savefig(PATH + 'rando_plot.png', dpi=300)
    plt.close()
    
    x = range(len(y))
    plt.plot(y.cumsum())
    plt.savefig(PATH + 'cumsum.png', dpi=300)
    plt.close()
    


if __name__ ==  "__main__":
    one_dims()