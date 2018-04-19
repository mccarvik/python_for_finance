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
# Multiple possible joint density functions for each combination of individual density functions
# So cant explictly get joint PDF from two single pdfs


def joint_dist_example(m1, m2, o1, o2):
    dens_funcs = []
    dens_func.append(lambda x: (1 / (np.sqrt(2 * pi * o1))) * np.exp(-(x**2) / (2 * o1)))
    dens_func.append(lambda x: (1 / (np.sqrt(2 * pi * o2))) * np.exp(-(x**2) / (2 * o2)))
    
    def cum_dist(low, hi):
        return integrate.quad(func, low, hi)[0]


def multivar_normal_dist(m, cov_mat):
    k = len(m)   # k = number of variables
    dens_func = lambda x: (1 /(np.sqrt(np.linalg.det(cov_mat) * (2*pi)**k))) * (np.exp((-1/2) * (x - m).T.dot(np.linalg.inv(cov_mat)).dot(x - m))) 
    
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
    plt.savefig('png/bivariate_dists/' + 'bivariate' + '.png', dpi=300)
    plt.close()


# Just doing this for 2 assets for now
def min_var_port(r, o, cov_mat, max_var):
    w1 = np.arange(0, 1.01, 0.05)
    w2 = 1 - w1
    wgts = zip(w1, w2)
    p = cov_mat[1,0] / (np.sqrt(cov_mat[0,0]) * np.sqrt(cov_mat[1,1]))
    opt = [0, 0, 0]
    for i in wgts:
        var = i[0]**2*o[0]**2 + i[1]**2*o[1]**2 + 2*i[0]*i[1]*p
        ret = i[0]*r[0] + i[1]*r[1]
        if var < max_var and ret > opt[0]:
            opt = [ret[0], var[0], i[0], i[1]]
            print(opt)
    return opt

if __name__ == '__main__':
    # print(joint_dist_example(0, 0.25, 0, 1))
    # cov_mat = np.array([[0.25, 0.30], [0.30, 1.00]])
    # cov_mat = np.array([[1, 0], [0, 1]])
    # m = np.array([[0], [0]])
    # multivar_normal_dist(m, cov_mat)
    
    cov_mat = np.array([[0.25, 0.30], [0.30, 1.00]])
    m = np.array([[0.1], [1.6]])
    o = np.array([[0.1], [1.6]])
    min_var_port(m, o, cov_mat, 2)