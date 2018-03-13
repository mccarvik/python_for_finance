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
    # Exponentially Weighted Moving Average model
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
    

def garch_model(data, lam=0.01, alpha=0.13, beta=0.86, long_run=0.0002):
    # Same as EWMA except with some weight given to the long run average
    # lambda + alpha + beta = 1
    daily_chg = data.pct_change(1)
    init_vol = 0.01
    vols = []
    for i in range(len(daily_chg)):
        if i == 0 or i == 1:
            continue
        if i == 2:
            # var = lam * long_run + alpha * init_vol**2 + beta * (0.016)**2
            var = lam * long_run + alpha * init_vol**2 + beta * daily_chg[i-1]**2
        else:
            var = lam * long_run + alpha *  vols[i-3]**2 + beta * daily_chg[i-1]**2
        vols.append(np.sqrt(var))
    return vols


def est_future_garch(long_run, alpha, beta, var, t):
    # forecasts the volatility on n + t using info available at n-1
    return long_run + (alpha + beta)**t * (var - long_run)
    

def vol_term_structure(long_run, alpha, beta, cur_var, T):
    # Calculates the estimated 1-year vol at time T, given the current variance
    # Can be used to estimate vol term structure
    # T measured in days
    a = np.log( 1 / (alpha + beta))
    est_var = 252 * (long_run + ((1 - np.exp(-a * T)) / (a * T)) * (cur_var - long_run))
    return est_var**0.5
    

def vol_change_impact(long_run, alpha, beta, cur_var, T, vol_chg):
    # Calculates the change to o(T) when o(0) changes
    # T measured in days
    # vol_chg measured in %, i.e. 0.01 = 100 bp move
    # interpretation: an instantaneous change of vol will result in the below chg in vol of term = t days 
    pdb.set_trace()
    a = np.log( 1 / (alpha + beta))
    oT = vol_term_structure(long_run, alpha, beta, cur_var, T)
    o0 = np.sqrt(252) * np.sqrt(cur_var)
    return ((1 - np.exp(-1*a*T)) / (a * T)) * (o0 / oT) * vol_chg * 100
    
    
def setup_data(data1, data2):
    data = pd.merge(data1, data2, how='inner', on='date')
    data['appl_dret'] = data['aapl'].pct_change(1).round(3)
    data['spy_dret'] = data['spy'].pct_change(1).round(3)  
    return data
    

def cov_estimate(data):
    # tot = 0
    # for ix, row in data.iterrows():
    #     tot += row['appl_dret'] * row['spy_dret']
    # return tot / len(data)
    
    # Can also use an EWMA model for updating covariances
    lam = 0.95
    corr = 0.6
    x_est_vol = 0.01
    y_est_vol = 0.02
    cov = corr * x_est_vol * y_est_vol
    x_chg = 0.005
    y_chg = 0.025
    # cov = lam * last_cov + (1 - lam) * x_chg * y_chg
    cov_upd = lam * cov + (1-lam) * x_chg * y_chg
    var_x_upd = lam * x_est_vol**2 + (1-lam) * x_chg**2
    var_y_upd = lam * y_est_vol**2 + (1-lam) * y_chg**2
    corr_upd = cov_upd / (np.sqrt(var_x_upd) * np.sqrt(var_y_upd))
    return corr_upd


if __name__ == '__main__':
    data1 = pd.read_csv('aapl.csv')
    data1.columns = ['date','aapl']
    # print(est_var(data1['aapl']))
    # print(est_var_daily_ret(data1['aapl']))
    # print(est_var_daily_ret_weights(data1['aapl']))
    # print(est_var_daily_ret_weights(data1['aapl'], [0.5, 0.3, 0.05], [1.1, 0.15]))
    # print(ewma_model(data1['aapl']))
    # print(garch_model(data1['aapl']))
    # print(est_future_garch(0.0002075, 0.13, 0.86, 0.0003, 10))
    
    #  for t in [10, 30, 50, 100, 500]:
    #     print(vol_term_structure(0.0002075, 0.1335, 0.86, 0.0003, t))
    #     print(vol_change_impact(0.0002075, 0.1335, 0.86, 0.0003, t, 0.01))
        
    data2 = pd.read_csv('spy.csv')
    data2.columns = ['date','spy']
    data = setup_data(data1, data2)
    print(cov_estimate(data))
    
    