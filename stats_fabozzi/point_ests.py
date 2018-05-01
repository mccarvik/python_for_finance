import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from math import sqrt, pi, log, e
from scipy.stats import bernoulli

# from dx import *
# from utils.utils import *

import numpy as np
import pandas as pd
import datetime as dt

# NOTES
# When taking samples for parameter estimation and plotting the samples, the results will be normally distributeds


def sample_mean(n, s):
    # get n smaples of size s and graph it
    vals = []
    for i in range(n):
        tot = 0
        for j in range(s):
            tot += np.random.normal()
        tot = tot / s
        vals.append(tot)
    plot_hist(vals, 'normal_mean_hist')
    

def sample_p_bern(n, s, p=0.5):
    # get n smaples of size s and graph it
    vals = []
    for i in range(n):
        tot = 0
        for j in range(s):
            tot += bernoulli.rvs(p)
        tot = tot / s
        vals.append(tot)
    plot_hist(vals, 'bernoulli_p_hist')


def sample_lam_poisson(n, s, p=0.5):
    # get n smaples of size s and graph it
    vals = []
    for i in range(n):
        tot = 0
        for j in range(s):
            tot += np.random.poisson()
        tot = tot / s
        vals.append(tot)
    plot_hist(vals, 'poisson_lam_hist')
    

def plot_hist(vals, title):
    # the histogram of the data
    n, bins, patches = plt.hist(vals, 30, normed=1)

    # add a 'best fit' line
    # mu, sigma = 0, 1
    # y = mlab.normpdf(bins, mu, sigma)
    # l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('mean')
    plt.ylabel('freq')
    plt.grid(True)
    plt.savefig('png/point_ests/' + title + '.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    # sample_mean(700, 10)
    # sample_p_bern(700, 10)
    sample_lam_poisson(700, 10)