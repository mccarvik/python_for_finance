import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# from dx import *
# from utils.utils import *

import numpy as np
import pandas as pd
import datetime as dt

PATH = '/home/ubuntu/workspace/python_for_finance/png/stats_fabozzi/'

def cum_freq_dist(data, x=50):
    tot = 0
    for ix, row in data.iterrows():
        if float(row[1]) <= x:
            tot += 1
    return tot / len(data)
    

# Rule for the optimal number of classes for a set of data
def sturges_rule(data):
    # return 1 + np.log(len(data))
    return int((1 + 3.222 * np.log10(len(data))) + 0.5)
    

# Rule for the optimal width of classes for a set of data
def freedman_diaconis_rule(data):
    q75, q25 = np.percentile(data[1], [75 ,25])
    iqr = q75 - q25
    pdb.set_trace()
    fd_width = (2 * iqr * len(data)**(-1/3))
    return int((data[1].max() - data[1].min()) / fd_width + 0.5)
    

def cum_freq_dist_graph(x, n_bins):
    fig, ax = plt.subplots(figsize=(8, 4))

    # plot the cumulative histogram
    n, bins, patches = ax.hist(x, n_bins, normed=1, histtype='step', cumulative=True, label='Empirical')
    
    # Add a line showing the expected distribution.
    # y = mpl.normpdf(bins, mu, sigma).cumsum()
    # y /= y[-1]
    
    ax.grid(True)
    # ax.legend(loc='right')
    ax.set_title('Cumulative step histograms')
    ax.set_xlabel('X Value')
    ax.set_ylabel('Likelihood of occurrence')
    ax.set_xlim([0, data[1].max()])
    ax.set_ylim([0, 1.02])
    
    plt.savefig(PATH + 'cum_freq_dist.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    data = pd.read_csv('dow.csv', header=None)
    
    # print(cum_freq_dist(data, 50))
    cats = sturges_rule(data[1])
    # cats = freedman_diaconis_rule(data)
    cum_freq_dist_graph(data[1], cats)
    