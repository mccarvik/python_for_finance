import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
# sys.path.append("/usr/local/lib/python3.4/dist-packages")
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

from dx import *
from utils.utils import *

import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as ss
from math import sqrt, pi, log, e
from hull_examples.bsm_model import *

# yields = [0.0025, 0.01, 0.015, 0.025]
# dates = [dt.datetime(2015, 1, 1), dt.datetime(2015, 7, 1), dt.datetime(2015, 12, 31), dt.datetime(2018, 12, 31)]
yields = [0.01, 0.0366, 0.04]
dates = [dt.datetime(2015,1,1), dt.datetime(2016,7,1), dt.datetime(2017,1,1)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))

# Use this to simplify equation reading for getting normal distribution cumulative distribution functions
N = ss.norm.cdf

def delta(S0, K, r, T, vol, div, otype='C'):
    # delta is equal to N(d1), represents how much the option price moves per move in underlying
    if otype == 'C':
        return N(d1(S0, K, r, T, vol, div))
    else:
        return N(d1(S0, K, r, T, vol, div)) - 1
        
def theta(S0, K, r, T, vol, div, otype='C'):
    # rate of change of the value of the portfolio with respect to the passage of time
    num = -S0 * N_(d1(S0, K, r, T, vol, div)) * vol
    den = 2 * sqrt(T)
    if otype == 'C':
        addition = r * K * np.exp(-r*T) * N(d2(S0, K, r, T, vol, div))
        return (num / den) - addition
    else:
        addition = r * K * np.exp(-r*T) * N(-d2(S0, K, r, T, vol, div))
        return (num / den) + addition

def N_(x):
    return (1 / sqrt(2 * pi)) * np.exp(-x**2 / 2)
    
    
def gamma(S0, K, r, T, vol, div):
    # rate of change of the value of delta with respect to the change in the underlying
    return N_(d1(S0, K, r, T, vol, div)) / (S0 * vol * sqrt(T))


if __name__ == '__main__':
    # print(delta(42, 40, 0.1, 0.5, 0.2, 0.0, "C"))
    # print(delta(42, 40, 0.1, 0.5, 0.2, 0.0, "P"))
    # print(theta(49, 50, 0.05, 0.3846, 0.2, 0.0, "C"))
    # print(theta(49, 50, 0.05, 0.3846, 0.2, 0.0, "P"))
    print(gamma(49, 50, 0.05, 0.3846, 0.2, 0.0))
    