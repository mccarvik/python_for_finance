import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from math import sqrt, pi, log, e

# from dx import *
# from utils.utils import *

import numpy as np
import pandas as pd
import datetime as dt

from continuous_prob_dists import distribution, plot_density_function, plot_cumulative_dist_func


def gen_extreme_val_dist(xi):
    # Gaussian distribution
    # Gumbel dist --> xi = 0
    # Frechet dist --> xi > 0
    # Weibull dist --> xi < 0
    m = lambda: 0
    s = lambda: 1
    
    def func(x):
        if xi == 0:
            return np.exp(-x) * np.exp(-np.exp(-x))
        else:
            # some values are undefined for these distributions, returning 0 works
            if xi < 0 and x > 2:
                return 0
            if xi > 0 and x < -2:
                return 0
            try:
                y = (1 + xi*x)**((-1/xi)-1) * np.exp(-(1 + xi*x)**(-1/xi))
                return y
            except Exception as e:
                # some values are undefined for these distributions, returning 0 works
                print(e)
                return 0
                
    dist = distribution(func, m, s)
    def cum_dist(low, hi):
        return integrate.quad(func, low, hi)[0]
    
    dist.cum_dist = cum_dist
    dist.name = "xi=" + str(xi)
    # plot_density_function([dist], 'normal_dist', [-4,4,100])
    # plot_cumulative_dist_func([dist], 'normal_cum_dist', [-4,4,100])
    return dist



if __name__ == '__main__':
    gumbel = gen_extreme_val_dist(0)
    frechet = gen_extreme_val_dist(0.5)
    weibull = gen_extreme_val_dist(-0.5)
    plot_density_function([gumbel, frechet, weibull], 'gen_extreme_val_dist', [-4,4,100])
    plot_cumulative_dist_func([gumbel, frechet, weibull], 'gen_extreme_val_cum_dist', [-4,4,100])
    # print(norm.mean())
    # print(norm.stdev())