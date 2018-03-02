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


def est_var(data):
    # unbiased estimate of variance rate per day: o**2
    tot = 0
    m = data.mean()
    for i in data:
        tot += (i - m)**2
    tot = (1 / (len(data) - 1)) * tot
    return tot
    
    
def est_var_daily_ret(data):
    # Some assumptions: mean = 0, len - 1 replace d by len
    tot = 0
    for i in range(len(data)):
        if i==len(data)-1:
            continue
        # times 100 to get in percent terms
        u = (data[i] - data[i+1]) / data[i+1] * 100
        tot += (u)**2
    tot = (1 / len(data)) * tot
    return tot


def est_var_daily_ret_weights(data, wgts=[0.5, 0.3, 0.15, 0.05], long_run=0):
    # weighted vols going back x amount of days with option long run average parameter
    # weights + long run weight (may be 0) must equal 1
    tot = 0
    # long run average vol * weight given to long run average
    # This is the Engle ARCH model
    if long_run:
       long_run = long_run[0] * long_run[1]
    
    for i in range(len(wgts)):
        # times 100 to get in percent terms
        u = wgts[i] * ((data[i] - data[i+1]) / data[i+1] * 100)**2
        tot += u
    return tot + long_run


def ewma_model(data, lam=0.9):
    # exponentially weighted moving average model
    # Need an entry vol
    daily_chg = data.pct_change(1)
    init_vol = 0.01
    vols = []
    for i in range(len(daily_chg)):
        if i == 0 or i == 1:
            continue
        if i == 2:
            var = lam * init_vol**2 + (1 - lam) * daily_chg[i-1]**2
        else:
            var = lam * vols[i-3]**2 + (1 - lam) * daily_chg[i-1]**2
        vols.append(np.sqrt(var))
    return vols
    


if __name__ == '__main__':
    data1 = pd.read_csv('aapl.csv')
    data1.columns = ['date','aapl']
    # print(est_var(data1['aapl']))
    # print(est_var_daily_ret(data1['aapl']))
    # print(est_var_daily_ret_weights(data1['aapl']))
    # print(est_var_daily_ret_weights(data1['aapl'], [0.5, 0.3, 0.05], [1.1, 0.15]))
    print(ewma_model(data1['aapl']))
    
    
    