import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.special as ss
from operator import mul
from functools import reduce

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


def hypergeometric_dist(N, K, n):
    dist = distribution([], [])
    for k in range(n+1):
        Kk = binom_coeff(K, k)
        Nn = binom_coeff(N, n)
        NKnk = binom_coeff(N-K, n-k)
        dist.vals.append(k)
        dist.weights.append((Kk * NKnk) / Nn)
    return dist


def multinomial_dist(N, ps):
    dist = distribution([], [])
    Xs = multinomial_combos(N, ps)
    for x in Xs:
        mc = multinomial_coeff(x)
        dist.vals.append(x)
        p_prod = reduce(mul, [p**i for p,i in zip(ps, x)], 1)
        dist.weights.append(p_prod * mc)
    
    dist.mean = lambda x: ps[x] * N
    dist.stdev = lambda x: np.sqrt(ps[x] * (1 - ps[x]) * N)
    return dist


def multinomial_combos(N, ps, id=0, combo=[]):
    if id+1 == len(ps):
        return N
    
    for i in range(N+1):
        ret = multinomial_combos(N-i, ps, id+1, combo)
        if type(ret) != list:
            combo.append([i, ret])
        else:
            combo = combo + [[i] + s for s in ret]
            combo = [c for c in combo if len(c) == len(ps)]
    return combo


def multinomial_coeff(ns):
    res, i = 1, 1
    for a in ns:
        for j in range(1, a+1):
            res *= i
            res //= j
            i += 1
    return res
    

def poisson_dist(N, lam, t):
    dist = distribution([], [])
    for k in range(N+1):
        dist.vals.append(k)
        wgt = ((lam * t)**k / np.math.factorial(k)) * np.exp(-lam * t)
        dist.weights.append(wgt)
    return dist


def disc_uni_dist(N):
    dist = distribution([], [])
    for k in range(1,N+1):
        dist.vals.append(k)
        dist.weights.append(1/N)
    pdb.set_trace()
    return dist
    

if __name__ == '__main__':
    # print(mean(dice_dist()))
    # print(mean(bernouli_dist(0.6, [22, 18])))
    # print(stdev(bernouli_dist(0.6, [22, 18])))
    
    # print(mean(binomial_dist(10, 0.5)))
    # print(stdev(binomial_dist(10, 0.5)))
    
    # print(mean(hypergeometric_dist(10, 4, 5)))
    # print(stdev(hypergeometric_dist(10, 4, 5)))
    
    # dist = multinomial_dist(10, [0.3, 0.3, 0.4])
    # print(dist.mean(0))
    # print(dist.stdev(0))
    
    # print(mean(poisson_dist(100, 2, 4)))
    # print(stdev(poisson_dist(100, 2, 4)))
    
    print(mean(disc_uni_dist(10)))
    print(stdev(disc_uni_dist(10)))
    
    
    
    
    
    
    
    