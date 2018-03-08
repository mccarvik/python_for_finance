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


def normal_dist(mean, stdev):
    m = lambda: mean
    s = lambda: stdev
    func = lambda x: (1 / (sqrt(2 * pi) * np.sqrt(stdev))) * np.exp(-1 * (x-mean)**2 / (2 * stdev))
    dist = distribution(func, m, s)
    
    def zscore(hi, low):
        return integrate.quad(func, hi, low)[0]
    
    dist.z_score = zscore
    plot_density_function(dist, 'normal_dist', [-4,4,100])
    return dist
    



if __name__ == '__main__':
    norm = normal_dist(0, 1)
    print(norm.mean())
    print(norm.stdev())
    print(norm.z_score(-3, 3))