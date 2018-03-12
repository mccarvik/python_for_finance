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


class distribution():
    def __init__(self, func, mean, stdev):
        self.func = func
        self.mean = mean
        self.stdev = stdev
        
def plot_density_function(dist, title, pts=[0,1,100]):
    xs = np.linspace(pts[0], pts[1], pts[2])
    ys = [dist.func(x) for x in xs]
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(xs, ys)
    plt.savefig('png/' + title+ '.png', dpi=300)
    plt.close()


def plot_cumulative_dist_func(dist, title, pts=[0,1,100]):
    xs = np.linspace(pts[0], pts[1], pts[2])
    ys = [dist.cum_dist(pts[0], x) for x in xs]
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(xs, ys)
    plt.savefig('png/' + title+ '.png', dpi=300)
    plt.close()


def normal_dist(mean, stdev):
    # Gaussian distribution
    m = lambda: mean
    s = lambda: stdev
    func = lambda x: (1 / (sqrt(2 * pi) * np.sqrt(stdev))) * np.exp(-1 * (x-mean)**2 / (2 * stdev))
    dist = distribution(func, m, s)
    
    def zscore(low, hi):
        return integrate.quad(func, low, hi)[0]
    
    dist.z_score = zscore
    dist.cum_dist = zscore
    plot_density_function(dist, 'normal_dist', [-4,4,100])
    plot_cumulative_dist_func(dist, 'normal_cum_dist', [-4,4,100])
    return dist
    

def chi_squared_dist(df=1):
    # The χ2n distribution is defined as the distribution that results from summing the squares of df independent random variables N(0,1)
    m = lambda: df
    s = lambda: np.sqrt(2*df)
    
    def gamma(x):
        # This only works for integers
        # return np.math.factorial(x-1)
        func = lambda t: np.exp(-t) * t**(x-1)
        return integrate.quad(func, 0, float('inf'))[0]
    
    def dens_func(x, df=df):
        if x < 0:
            return 0
        else:
            return ((1 / (2**(df/2) * gamma(df/2))) * np.exp(-x/2) * x**(df/2 - 1))

    dist = distribution(dens_func, m, s)
    
    def cum_dist(low, hi):
        return integrate.quad(dens_func, low, hi)[0]
    
    dist.cum_dist = cum_dist
    plot_density_function(dist, 'chi_squared_dist', [0,15,100])
    plot_cumulative_dist_func(dist, "chi_squared_cum_dist", [0,15,100])
    return dist


def students_t(df=3):
    # arises when estimating the mean of a normally distributed population in situations where the sample size is small 
    # and population standard deviation is unknown
    m = lambda: 0
    s = lambda: np.sqrt(df / (df-2))
    
    def gamma(x):
        # This only works for integers
        # return np.math.factorial(x-1)
        func = lambda t: np.exp(-t) * t**(x-1)
        return integrate.quad(func, 0, float('inf'))[0]
    
    def dens_func(x, df=df):
        front = 1 / (np.sqrt(df*pi))
        mid = gamma((df+1)/2) / gamma(df/2)
        back = (1 + (x**2 / df))**(-1 * ((df+1) / 2))
        return front * mid * back

    dist = distribution(dens_func, m, s)
    
    def cum_dist(low, hi):
        return integrate.quad(dens_func, low, hi)[0]
    
    dist.cum_dist = cum_dist
    plot_density_function(dist, 'students_t_dist', [-4,4,100])
    plot_cumulative_dist_func(dist, "students_t_cum_dist", [-4,4,100])
    return dist    



if __name__ == '__main__':
    # norm = normal_dist(0, 1)
    # print(norm.mean())
    # print(norm.stdev())
    # print(norm.z_score(-1, 1))
    
    # x_sqr = chi_squared_dist(5)
    # print(x_sqr.mean())
    # print(x_sqr.stdev())
    
    students_t = students_t(3)
    print(students_t.mean())
    print(students_t.stdev())