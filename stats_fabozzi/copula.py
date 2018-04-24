import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
from math import sqrt, pi, log, e

# from dx import *
# from utils.utils import *

import numpy as np
import pandas as pd
import datetime as dt

# NOTES
# need to study this more, very difficult concept


def gaussian_copula(p):
    # d = 2, only 2 distributions to simplify the calculations
    cov_mat = np.array([[1,00, p], [p, 1.00]])
    
    # Need to do multiple integration
    # https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html
    
    def dens_func(x1, x2):
        front = 1 / (2 * pi * np.sqrt(1 - p**2))
        exp = (x1**2 - 2 * p * x1 * x2 + x2**2)
    
    dens_func = lambda x: (1 /(np.sqrt(np.linalg.det(cov_mat) * (2*pi)**k))) * (np.exp((-1/2) * (x - m).T.dot(np.linalg.inv(cov_mat)).dot(x - m))) 
    
    xs = []
    for i in range(k):
      xs.append(np.arange(0, 1, 0.05))
    
    # Rest only works for bivariate, more than 2 vairables wont work
    X1, X2 = np.meshgrid(xs[0], xs[1])
    Z = np.zeros(shape=(len(X1), len(X2)))
    R = np.sqrt(X1**2 + X2**2)
    for i in range(len(xs[0])):
        for j in range(len(xs[1])):
            x = np.array([[xs[0][i]],[xs[1][j]]])
            Z[i,j] = dens_func(x)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X1, X2, Z, linewidth=0, antialiased=False)
    plt.savefig('png/copulas/' + 'gaussian_' + str(p) + '.png', dpi=300)
    plt.close()
    

def multivar_tstudent_dist(m, cov_mat, df):
    k = len(m)   # k = number of variables
    v = df       # degress of freedom
    
    def gamma(x):
        # This only works for integers
        # return np.math.factorial(x-1)
        func = lambda t: np.exp(-t) * t**(x-1)
        return integrate.quad(func, 0, float('inf'))[0]
    
    def dens_func(x):
        front = gamma((1/2) * (v + k)) / (gamma((1/2)*v) * (pi * v)**(k/2) * np.linalg.det(cov_mat)**(1/2))
        back = (1 + ((x - m).T.dot(np.linalg.inv(cov_mat)).dot(x - m)) / v)**(-(v+k)/2)
        return front * back
    
    def cum_dist(low, hi):
        return integrate.quad(dens_func, low, hi)[0]
    
    xs = []
    for i in range(k):
      xs.append(np.arange(-3, 3.01, 0.05))
    
    # Rest only works for bivariate, more than 2 vairables wont work
    X1, X2 = np.meshgrid(xs[0], xs[1])
    Z = np.zeros(shape=(len(X1), len(X2)))
    R = np.sqrt(X1**2 + X2**2)
    for i in range(len(xs[0])):
        for j in range(len(xs[1])):
            x = np.array([[xs[0][i]],[xs[1][j]]])
            Z[i,j] = dens_func(x)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X1, X2, Z, linewidth=0, antialiased=False)
    plt.savefig('png/bivariate_dists/' + 'bivariate_tstud' + '.png', dpi=300)
    plt.close()



if __name__ == '__main__':
    gaussian_copula(0)
    # gaussian_copula(0.5)
    # gaussian_copula(0.95)
    
    # cov_mat = np.array([[0.25, 0.30], [0.30, 1.00]])
    # m = np.array([[0.1], [1.6]])
    # o = np.array([[0.1], [1.6]])
    # min_var_port(m, o, cov_mat, 2)
    