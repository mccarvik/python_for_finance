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


# NOTES
# Second central moment = variance
# third central moment = skew
# fourth central moment = kurtosis


class distribution():
    def __init__(self, func, mean, stdev):
        self.func = func
        self.mean = mean
        self.stdev = stdev
 
   
def plot_density_function(dists, title, pts=[0,1,100]):
    xs = np.linspace(pts[0], pts[1], pts[2])
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    mn = float("inf")
    mx = float("-inf")
    for dist in dists:
        ys = [dist.func(x) for x in xs]
        plt.plot(xs, ys, label=dist.name)
        mn = min(mn, min(ys))
        mx = max(mx, max(ys))
    
    mn = mn * 0.95 if mn > 0 else mn * 1.05
    mx = mx * 1.05 if mx > 0 else mx * 0.95
    plt.ylim(mn, mx)
    plt.legend(loc='upper left')
    plt.savefig('png/cont_dists/' + title + '.png', dpi=300)
    plt.close()


def plot_cumulative_dist_func(dists, title, pts=[0,1,100]):
    xs = np.linspace(pts[0], pts[1], pts[2])
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    mn = float("inf")
    mx = float("-inf")
    for dist in dists:
        ys = [dist.cum_dist(pts[0], x) for x in xs]
        plt.plot(xs, ys, label=dist.name)
        mn = min(mn, min(ys))
        mx = max(mx, max(ys))
    
    mn = mn * 0.95 if mn > 0 else mn * 1.05
    mx = mx * 1.05 if mx > 0 else mx * 0.95
    plt.ylim(mn, mx)
    plt.legend(loc='lower right')
    plt.savefig('png/cont_dists/' + title + '.png', dpi=300)
    plt.close()


def normal_dist(mean, stdev):
    # Gaussian distribution
    m = lambda: mean
    s = lambda: stdev
    func = lambda x: (1 / (sqrt(2 * pi) * np.sqrt(stdev))) * np.exp(-1 * (x-mean)**2 / (2 * stdev))
    dist = distribution(func, m, s)
    dist.skewness = lambda: 0
    dist.kurtosis = lambda: 3
    
    def zscore(low, hi):
        return integrate.quad(func, low, hi)[0]
    
    dist.z_score = zscore
    dist.cum_dist = zscore
    dist.name = "m=" + str(mean) + " s=" + str(stdev)
    # plot_density_function([dist], 'normal_dist', [-4,4,100])
    # plot_cumulative_dist_func([dist], 'normal_cum_dist', [-4,4,100])
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
    dist.skewness = lambda: np.sqrt(8/df)
    dist.kurtosis = lambda: 3 + 12/df
    
    def cum_dist(low, hi):
        return integrate.quad(dens_func, low, hi)[0]
    
    dist.cum_dist = cum_dist
    # plot_density_function(dist, 'chi_squared_dist', [0,15,100])
    # plot_cumulative_dist_func(dist, "chi_squared_cum_dist", [0,15,100])
    return dist


def students_t_dist(df=3):
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
    dist.skewness = lambda: 0   # if n >= 4
    dist.kurtosis = lambda: 3 + 6 / (df - 4)    # if df >= 5
    
    def cum_dist(low, hi):
        return integrate.quad(dens_func, low, hi)[0]
    
    dist.cum_dist = cum_dist
    plot_density_function(dist, 'students_t_dist', [-4,4,100])
    plot_cumulative_dist_func(dist, "students_t_cum_dist", [-4,4,100])
    return dist


def f_dist(n1, n2):
    # main use of F-distribution is to test whether two independent samples have been drawn for the normal populations with the same variance
    # or if two independent estimates of the population variance are homogeneous or not, since it is often desirable to compare two variances rather than two averages
    m = lambda: n2 / (n2 - 2)
    s = lambda: np.sqrt((2 * n2**2 * (n1 + n2 -2)) / (n1 * (n2 -2)**2 * (n2 - 4)))
    
    def gamma(x):
        # This only works for integers
        # return np.math.factorial(x-1)
        func = lambda t: np.exp(-t) * t**(x-1)
        return integrate.quad(func, 0, float('inf'))[0]
    
    def beta(x, y):
        return (gamma(x) * gamma(y)) / gamma(x+y)
    
    def dens_func(x):
        c = (n1 / n2)**(n1/2) * (1 / beta(n1/2, n2/2))
        if x < 0:
            return 0
        else:
            return c * x**(n1/2 - 1) * (1 + (n1/n2) * x)**(-1 * (n1 + n2) / 2)

    dist = distribution(dens_func, m, s)
    dist.skewness = lambda: ((2*n1 + n2 - 2) * (np.sqrt(8 * (n2-4)))) / ((n2 - 6) * np.sqrt(n1 * (n1 + n2 - 2)))    # n2 > 6
    dist.kurtosis = lambda: 12 * ((n1 * (5*n2 - 22) * (n1 + n2 -2) + (n2 - 4) * (n2 - 2)**2) / (n1 * (n2 - 6) * (n2 - 8) *(n1 + n2 - 2)))   # n2 > 8
    
    def cum_dist(low, hi):
        return integrate.quad(dens_func, low, hi)[0]
    
    dist.cum_dist = cum_dist
    plot_density_function(dist, 'f_dist', [0,5,100])
    plot_cumulative_dist_func(dist, "f_cum_dist", [0,5,100])
    return dist
    

def exponential_dist(lam):
    # describes the time between events in a Poisson point process
    # i.e., a process in which events occur continuously and independently at a constant average rate rather than two averages
    m = lambda: 1 / lam
    s = lambda: np.sqrt(1 / lam**2)
    
    def dens_func(x):
        if x < 0:
            return 0
        else:
            return lam * np.exp(-1 * lam * x)

    dist = distribution(dens_func, m, s)
    dist.skewness = lambda: 2
    dist.kurtosis = lambda: 9
    
    def cum_dist(low, hi):
        return integrate.quad(dens_func, low, hi)[0]
    
    dist.cum_dist = cum_dist
    plot_density_function(dist, 'exp_dist', [0,5,100])
    plot_cumulative_dist_func(dist, "exp_cum_dist", [0,5,100])
    return dist


def rect_dist(a, b):
    m = lambda: (a + b) / 2
    s = lambda: np.sqrt((b - a)**2 / 12)
    
    def dens_func(x):
        if a <= x and x <= b:
            return 1 / (b - a)
        else:
            return 0
        
    dist = distribution(dens_func, m, s)
    dist.skewness = lambda: 0
    dist.kurtosis = lambda: 1.8
    
    def cum_dist(low, hi):
        return integrate.quad(dens_func, low, hi)[0]
    
    dist.cum_dist = cum_dist
    # plot_density_function(dist, 'rect_dist', [0,4,100])
    # plot_cumulative_dist_func(dist, "rect_cum_dist", [0,4,100])
    return dist


def gamma_dist(c=1, lam=1):
    # can be thought of as a waiting time between Poisson distributed events
    # The waiting time until the cth Poisson event with a rate of change λ is
    m = lambda: c / lam
    s = lambda: np.sqrt(c / lam**2)
    
    def gamma(x):
        # This only works for integers
        # return np.math.factorial(x-1)
        func = lambda t: np.exp(-t) * t**(x-1)
        return integrate.quad(func, 0, float('inf'))[0]
    
    def dens_func(x):
        if x < 0:
            return 0
        else:
            return (lam * (lam * x)**(c-1) * np.exp(-1 * lam * x)) / gamma(c)

    dist = distribution(dens_func, m, s)
    dist.skewness = lambda: 2 / np.sqrt(c)
    dist.kurtosis = lambda: 3 + 6/c
    
    def cum_dist(low, hi):
        return integrate.quad(dens_func, low, hi)[0]
    
    dist.cum_dist = cum_dist
    dist.name = "c=" + str(c) + " lam=" + str(lam)
    # plot_density_function([dist], 'gamma_dist', [0,5,100])
    # plot_cumulative_dist_func([dist], "gamma_cum_dist", [0,5,100])
    return dist


def erlang_dist(c=1, lam=1):
    # developed to find the number of phone calls which can be made simultaneously to switching station operators
    # if the shape parameter c is 1, the distribution is the same as the exponential distribution
    # Generalized form of the gamma distribution
    m = lambda: c / lam
    s = lambda: np.sqrt(c / lam**2)
    
    def dens_func(x):
        if x < 0:
            return 0
        else:
            return (lam**c * x**(c-1) * np.exp(-1 * lam * x)) / np.math.factorial(c - 1)
            # return 1 - (np.exp(-lam * x) * sum([(lam * x)**i / np.math.factorial(i) for i in range(c)]))

    dist = distribution(dens_func, m, s)
    
    def cum_dist(low, hi):
        return integrate.quad(dens_func, low, hi)[0]
    
    dist.cum_dist = cum_dist
    dist.name = "c=" + str(c) + " lam=" + str(lam)
    # plot_density_function([dist], 'gamma_dist', [0,5,100])
    # plot_cumulative_dist_func([dist], "gamma_cum_dist", [0,5,100])
    return dist


def beta_dist(c=1, d=1):
    # used to model things that have a limited range, like 0 to 1.
    # Examples are the probability of success in an experiment having only two outcomes, like success and failure
    # If you do a limited number of experiments, and some are successful, you can represent what that tells you by a beta distribution
    m = lambda: c / (c+d)
    s = lambda: np.sqrt((c*d) / ((c + d)**2 * (c + d + 1)))
    
    def gamma(x):
        # This only works for integers
        # return np.math.factorial(x-1)
        func = lambda t: np.exp(-t) * t**(x-1)
        return integrate.quad(func, 0, float('inf'))[0]
    
    def beta(x, y):
        return (gamma(x) * gamma(y)) / gamma(x+y)
    
    def dens_func(x):
        if x < 0:
            return 0
        elif x > 1:
            return 0
        else:
            return (1 / beta(c, d)) * x**(c-1) * (1-x)**(d-1)

    dist = distribution(dens_func, m, s)
    dist.skewness = lambda: 0   # complicated calc
    dist.kurtosis = lambda: (3 * (c + d + 1) * (c**2 * (d+2) + d**2 * (c+2) - 2*c*d)) / (c*d * (c+d+2) *(c+d+3))
    
    def cum_dist(low, hi):
        return integrate.quad(dens_func, low, hi)[0]
    
    dist.cum_dist = cum_dist
    dist.name = "c=" + str(c) + " d=" + str(d)
    # plot_density_function([dist], 'beta_dist', [0,5,100])
    # plot_cumulative_dist_func([dist], "beta_cum_dist", [0,5,100])
    return dist


def lognormal_dist(mean=0, stdev=1):
    # A random variable is lognormally distributed if its logarithm is normally distributed
    m = lambda: np.exp(mean + stdev**2 / 2)
    s = lambda: np.exp(stdev**2) * (np.exp(stdev**2) - 1) * np.exp(2*mean)
    
    
    def dens_func(x):
        if x > 0:
            return (1 / (x * stdev * np.sqrt(2*pi))) * np.exp(-1 * (np.log(x-mean)**2) / (2 * stdev**2))
        else:
            return 0

    dist = distribution(dens_func, m, s)
    dist.skewness = lambda: (np.exp(stdev**2) + 2) * np.sqrt(np.exp(stdev**2) - 1)
    dist.kurtosis = lambda: np.exp(stdev**2)**4 + 2 * np.exp(stdev**2)**3 + 3 * np.exp(stdev**2)**2 - 3
    
    def cum_dist(low, hi):
        return integrate.quad(dens_func, low, hi)[0]
    
    dist.cum_dist = cum_dist
    dist.name = "m=" + str(mean) + " s=" + str(stdev)
    # plot_density_function([dist], 'beta_dist', [0,5,100])
    # plot_cumulative_dist_func([dist], "beta_cum_dist", [0,5,100])
    return dist



if __name__ == '__main__':
    # norm = normal_dist(0, 1)
    # norm2 = normal_dist(0, 2)
    # plot_density_function([norm, norm2], 'normal_dist', [-4,4,100])
    # plot_cumulative_dist_func([norm, norm2], 'normal_cum_dist', [-4,4,100])
    # print(norm.mean())
    # print(norm.stdev())
    # print(norm.z_score(-1, 1))
    
    x_sqr = chi_squared_dist(5)
    print(x_sqr.mean())
    print(x_sqr.stdev())
    print(x_sqr.skewness())
    print(x_sqr.kurtosis())
    
    # students_t = students_t_dist(3)
    # print(students_t.mean())
    # print(students_t.stdev())
    
    # f_d = f_dist(4, 10)
    # print(f_d.mean())
    # print(f_d.stdev())
    
    # exp = exponential_dist(2)
    # print(exp.mean())
    # print(exp.stdev())
    
    # rect = rect_dist(1,3)
    # print(rect.mean())
    # print(rect.stdev())
    
    # gam = gamma_dist(1,1)
    # gam2 = gamma_dist(2,2)
    # gam3 = gamma_dist(4,4)
    # plot_density_function([gam, gam2, gam3], 'gamma_dist', [0,5,100])
    # plot_cumulative_dist_func([gam, gam2, gam3], 'gamma_cum_dist', [0,5,100])
    # print(gam.mean())
    # print(gam.stdev())
    
    # erlang1 = erlang_dist(1,1)
    # erlang2 = erlang_dist(5,1)
    # erlang3 = erlang_dist(9,1)
    # plot_density_function([erlang1, erlang2, erlang3], 'erlang_dist', [0,20,100])
    # plot_cumulative_dist_func([erlang1, erlang2, erlang3], 'erlang_cum_dist', [0,20,100])
    # print(erlang1.mean())
    # print(erlang1.stdev())
    
    # beta1 = beta_dist(1,1)
    # beta2 = beta_dist(5,5)
    # beta3 = beta_dist(1,4)
    # beta4 = beta_dist(4,1)
    # plot_density_function([beta1, beta2, beta3, beta4], 'beta_dist', [0,1,100])
    # plot_cumulative_dist_func([beta1, beta2, beta3, beta4], 'beta_cum_dist', [0,1,100])
    # print(beta1.mean())
    # print(beta1.stdev())
    
    # ln1 = lognormal_dist(0,1)
    # ln2 = lognormal_dist(0,0.5)
    # ln3 = lognormal_dist(0,2)
    # plot_density_function([ln1, ln2, ln3], 'ln_dist', [-1,3,100])
    # plot_cumulative_dist_func([ln1, ln2, ln3], 'ln_cum_dist', [-1,3,100])
    # print(ln1.mean())
    # print(ln1.stdev())
    
    