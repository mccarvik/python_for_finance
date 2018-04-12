import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sympy as sym
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
    def gamma(x):
            # This only works for integers
            # return np.math.factorial(x-1)
            func = lambda t: np.exp(-t) * t**(x-1)
            return integrate.quad(func, 0, float('inf'))[0]
    
    if xi >= 1:
        m = lambda: float('inf')
    elif xi == 0:
        # Euler–Mascheroni constant 
        m = lambda: 0.577215664901532860606512090082402431
    else:
        m = lambda: gamma(1 - xi) / xi
        
    if xi == 0:
        s = lambda: np.sqrt(pi**2 / 6)
    elif xi >= 1/2:
        s = lambda: float('inf')
    else:
        s = lambda: np.sqrt((gamma(1 - 2*xi) - gamma(1-xi)**2) / xi**2)
    
    
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


def gen_pareto_dist(xi, sigma, mu):
    # Distribution employed to model large values wich changes well beyond the typical change
    m = lambda: mu + (sigma / (1 - xi))
    s = lambda: np.sqrt(sigma**2 / ((1 - xi)**2 * (1 - 2 * xi)))
    
    def func(x):
        if xi == 0:
            # Need the derivative of this
            # can (maybe) just choose really small number for xi to approach this
            f = lambda y: (1 - np.exp(-1 * ((y - mu) / sigma)))
        else:
            return (1 / sigma) * (1 + xi * ((x - mu) / sigma))**(-1-(1/xi))
                
    dist = distribution(func, m, s)
    def cum_dist(low, hi):
        return integrate.quad(func, low, hi)[0]
    
    dist.cum_dist = cum_dist
    dist.name = "xi=" + str(xi) + " sigma=" + str(sigma)
    # plot_density_function([dist], 'normal_dist', [-4,4,100])
    # plot_cumulative_dist_func([dist], 'normal_cum_dist', [-4,4,100])
    return dist


if __name__ == '__main__':
    # gumbel = gen_extreme_val_dist(0)
    # frechet = gen_extreme_val_dist(0.5)
    # weibull = gen_extreme_val_dist(-0.5)
    # plot_density_function([gumbel, frechet, weibull], 'gen_extreme_val_dist', [-4,4,100])
    # plot_cumulative_dist_func([gumbel, frechet, weibull], 'gen_extreme_val_cum_dist', [-4,4,100])
    # print(weibull.mean())
    # print(weibull.stdev())
    
    pareto1 = gen_pareto_dist(-0.25, 1, 0)
    pareto2 = gen_pareto_dist(0.001, 1, 0)
    pareto3 = gen_pareto_dist(1, 1, 0)
    plot_density_function([pareto1, pareto2, pareto3], 'pareto_dist', [0,5,100])
    plot_cumulative_dist_func([pareto1, pareto2, pareto3], 'pareto_cum_dist', [0,5,100])
    print(pareto1.mean())
    print(pareto1.stdev())
    
    
    