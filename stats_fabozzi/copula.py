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
    # p = correlation between 2 variables
    # d = 2, only 2 distributions to simplify the calculations
    d = 2
    cov_mat = np.array([[1,00, p], [p, 1.00]])
    
    # Need to do multiple integration
    # https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html
    
    def dens_func(x1, x2):
        front = 1 / (2 * pi * np.sqrt(1 - p**2))
        exp = (x1**2 - 2 * p * x1 * x2 + x2**2) / (2 * (1 - p**2))
        return front * np.exp(exp)
    
    xs = []
    for i in range(d):
      xs.append(np.arange(0.001, 1, 0.05))
    
    # Rest only works for bivariate, more than 2 vairables wont work
    X1, X2 = np.meshgrid(xs[0], xs[1])
    Z = np.zeros(shape=(len(X1), len(X2)))
    R = np.sqrt(X1**2 + X2**2)
    for i in range(len(xs[0])):
        for j in range(len(xs[1])):
            # Z[i,j] = integrate.nquad(dens_func, [[0, xs[0][i]], [0, xs[1][j]]])
            Z[i,j] = integrate.dblquad(dens_func, 0, xs[0][i], lambda x: 0, lambda x: xs[1][j])[0]
    
    pdb.set_trace()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X1, X2, Z, linewidth=0, antialiased=False)
    plt.savefig('png/copulas/' + 'gaussian_' + str(p) + '.png', dpi=300)
    plt.close()
    

def multivar_tstudent_dist(p, v):
    # p = correlation between 2 variables
    # v = degress of freedom
    
    def gamma(x):
        # This only works for integers
        # return np.math.factorial(x-1)
        func = lambda t: np.exp(-t) * t**(x-1)
        return integrate.quad(func, 0, float('inf'))[0]
    
    # d = 2, only 2 distributions to simplify the calculations
    d = 2
    cov_mat = np.array([[1.00, p], [p, 1.00]])
    front = gamma(v/2 + 1) / (gamma(v/2) * np.sqrt(np.linalg.det(cov_mat) * (v * pi)**2))
    
    # Need to do multiple integration
    # https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html
    
    def dens_func(x1, x2):
        y = np.array([[x1],[x2]])
        return (1 + (y.T.dot(np.linalg.inv(cov_mat)).dot(y) / v))**(-1 * (v/2 + 1))
    
    xs = []
    for i in range(d):
      xs.append(np.arange(0.001, 1, 0.05))
    
    # Rest only works for bivariate, more than 2 vairables wont work
    X1, X2 = np.meshgrid(xs[0], xs[1])
    Z = np.zeros(shape=(len(X1), len(X2)))
    R = np.sqrt(X1**2 + X2**2)
    for i in range(len(xs[0])):
        for j in range(len(xs[1])):
            # Z[i,j] = integrate.nquad(dens_func, [[0, xs[0][i]], [0, xs[1][j]]])
            Z[i,j] = front * integrate.dblquad(dens_func, 0, xs[0][i], lambda x: 0, lambda x: xs[1][j])[0]
    
    pdb.set_trace()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X1, X2, Z, linewidth=0, antialiased=False)
    plt.savefig('png/copulas/' + 't_student_' + str(p) + '.png', dpi=300)
    plt.close()



if __name__ == '__main__':
    # gaussian_copula(0)
    # gaussian_copula(0.5)
    # gaussian_copula(0.95)
    
    multivar_tstudent_dist(0, 5)
    multivar_tstudent_dist(0.5, 5)
    multivar_tstudent_dist(0.95, 5)
    
    