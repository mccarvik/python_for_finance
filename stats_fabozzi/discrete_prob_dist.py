import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.special as ss

# from dx import *
# from utils.utils import *

import numpy as np
import pandas as pd
import datetime as dt


class distribution():
    def __init__(self, vals, weights):
        self.weights = weights
        self.vals = vals


def mean(dist):
    tot_val = 0
    for v, w in zip(dist.vals, dist.weights):
        tot_val += w * v
    return tot_val


def stdev(dist):
    m = mean(dist)
    pdb.set_trace()
    var = sum([(v - m)**2 * w for v, w in zip(dist.vals, dist.weights)])
    return np.sqrt(var)
    

def dice_dist():
    dist = distribution([], [])
    for i in range(1, 7):
        dist.vals.append(i)
        dist.weights.append(1/6)
    return dist
    

def bernouli_dist(p, v):
    dist = distribution([], [])
    dist.vals.append(v[0])
    dist.weights.append(p)
    dist.vals.append(v[1])
    dist.weights.append(1-p)
    return dist


def binom_coeff(n, k):
    return ss.binom(n, k)


def binomial_dist(n, p):
    dist = distribution([], [])
    for i in range(n+1):
        dist.vals.append(i)
        w = binom_coeff(n, i) * p**i * (1-p)**(n-i)
        dist.weights.append(w)
    return dist
    

if __name__ == '__main__':
    # print(mean(dice_dist()))
    # print(mean(bernouli_dist(0.6, [22, 18])))
    # print(stdev(bernouli_dist(0.6, [22, 18])))
    
    print(mean(binomial_dist(10, 0.5)))
    print(stdev(binomial_dist(10, 0.5)))