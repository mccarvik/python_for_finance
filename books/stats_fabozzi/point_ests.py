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
# MSE = mean squared error
# MSE = Var(x) + (E(x) - x)**2
# Sample mean is an unbiased estimator so E(x) = x and MSE_mean is just = var(x)
# MSE measures spread of parameter estimator
# Tradeoff bewtween variance (standard error) and bias, decreasing one increases the other
# An estimator is consistent if as the sample size increases it gets more accurate
# An estimator A is more efficient than an estimator B if it has less variance
# Maximum Likelihood Estimator (MLE) - yields the paramter value with the greatest likelihood of the given observation x
# Cramer-Rao Lower Bound - minimum bound that no estimator variance will ever fall below
#   based on the second derivative of tbe log-likelihood function


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


def sample_var_bias(samps):
    # the variance of a sample variance is biased, biased disappears with larger amount of values pulled for the sample
    samp_m = sum(samps) / len(samps)
    sample_var = (1 / len(samps)) * sum([(x - samp_m)**2 for x in samps])
    sample_var_bias_adj = (1 / (len(samps) - 1)) * sum([(x - samp_m)**2 for x in samps])
    return (sample_var, sample_var_bias_adj)


def MSE_var(samps):
    size = len(samps)
    o = np.std(samps)
    return ((size - 1) / size) * ((2*o**4) / (size - 1)) * (1 / size)**2 * o**4

    
def effic_sample_var(samps):
    n = len(samps)
    o = np.std(samps)
    est1 = np.array([[o**2/n, 0], [0, ((n-1)/n)**2 * ((2*o**2)/(n-1))]])     # bias adjusted
    est2 = np.array([[o**2/n, 0], [0, ((2*o**2)/(n-1))]])  # simple sample variance
    pos_semidef_cov_mat = est1 - est2
    print(pos_semidef_cov_mat)




if __name__ == '__main__':
    rand_sam = []
    n = 10
    for i in range(n):
        rand_sam.append(np.random.normal())
        
    # sample_mean(700, 10)
    # sample_p_bern(700, 10)
    # sample_lam_poisson(700, 10)
    # print(sample_var_bias(rand_sam))
    # print(np.std(rand_sam)**2)
    # print(MSE_var(rand_sam))
    print(effic_sample_var(rand_sam))