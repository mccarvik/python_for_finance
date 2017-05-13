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
    
    x = range(len(y))
    plt.plot(y.cumsum())
    plt.grid(True)
    plt.axis('tight')
    plt.xlim(-1, 20)
    plt.ylim(np.min(y.cumsum()) - 1,
            np.max(y.cumsum()) + 1)
    plt.savefig(PATH + 'cumsum_grid.png', dpi=300)
    plt.close()
    
def two_dims():
    np.random.seed(2000)
    y = np.random.standard_normal((20,2)).cumsum(axis=0)
    y[:, 0] = y[:, 0] * 100
    # key line, gives us axis to the figure and axises separately
    fig, ax1 = plt.subplots()
    # plt.figure(figsize=(7,4))
    plt.plot(y[:, 0], 'b', lw=1.5, label='1st')
    # ro = red dots
    plt.plot(y[:, 0], 'ro')
    plt.legend(loc=8)
    plt.axis('tight')
    plt.xlabel('index')
    plt.ylabel('value 1st')
    plt.title('A Simple Plot')
    
    # Key line below - get a second plot that shares the x axis
    ax2 = ax1.twinx()
    plt.plot(y[:,1], 'g', lw=1.5, label='2nd')
    plt.plot(y[:, 1], 'ro')
    plt.legend(loc=0)
    plt.ylabel('value 2nd')
    # plt.grid(True)
    plt.savefig(PATH + 'two_dim_grid.png', dpi=300)
    plt.close()

def sep_plots():
    np.random.seed(2000)
    y = np.random.standard_normal((20,2)).cumsum(axis=0)
    y[:, 0] = y[:, 0] * 100
    plt.figure(figsize=(7,5))
    plt.subplot(211)
    plt.plot(y[:, 0], lw=1.5, label='1st')
    plt.plot(y[:,0], 'ro')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.ylabel('value')
    plt.title('A Simple Plot')
    plt.subplot(212)
    plt.plot(y[:, 1], lw=1.5, label='2nd')
    plt.plot(y[:, 1], 'ro')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.xlabel('index')
    plt.ylabel('value')
    plt.savefig(PATH + 'sep_plots.png', dpi=300)
    plt.close()
    

if __name__ ==  "__main__":
    sep_plots()
    

