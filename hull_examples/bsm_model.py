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

# yields = [0.0025, 0.01, 0.015, 0.025]
# dates = [dt.datetime(2015, 1, 1), dt.datetime(2015, 7, 1), dt.datetime(2015, 12, 31), dt.datetime(2018, 12, 31)]
yields = [0.01, 0.0366, 0.04]
dates = [dt.datetime(2015,1,1), dt.datetime(2016,7,1), dt.datetime(2017,1,1)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))


def lognormal_mean(underlying_px, exp_ret, vol, T):
    # assume percentage changes in stock price in a very short period of time are normally distributed
    # This means the stock price at the end of time T will be lognormally distributed
    # This function returns the mean and std_dev of the natural log of ending stock price (ln St) which is normally distributed
    #       - i.e. if ln St is normally distributed, St is lognormally distributed
    return (np.log(underlying_px) + (exp_ret - vol**2/2) * T, vol**2 * T)


def lognormal_exp_ret_vol(S0, exp_ret, vol, T):
    # lognormal expected return and standard dev
    # E(St) = S0e^(uT)
    # var(St) = S0^2*e^(2uT)*(e^(o^2*T)-1)
    ret = S0 * np.exp(exp_ret * T)
    var = S0**2 * np.exp(2*exp_ret*T) * (np.exp(vol**2 * T) - 1)
    st_dev = np.sqrt(var)
    return (ret, st_dev)


def distribution_rate_of_ret(exp_ret, vol, T):
    # The continuously compounded rate of return is normally dsitributed
    mean = exp_ret - vol**2 / 2
    st_dev = vol / np.sqrt(T)
    return (mean, st_dev)
    

def get_daily_return(data, divs=None):
    # can incorporate dividends by adding the total value of the dividend to the stock on the day it goes ex-div
    ret = []
    for i in range(len(data)):
        if i==0:
            last = data[i]
            continue
        if not divs:
            ret.append(np.log(data[i] / last))
        else:
            ret.append(np.log((data[i] + divs[i]) / last))
        last = data[i]
    return ret


def estimating_vol(data, t):
    m = np.mean(data)
    # s = np.sqrt( (1 / (len(data)-1))  * sum([(u-m)**2 for u in data['Close']]))
    # or, equations reuslt in same estimation
    s = np.sqrt( (1 / (len(data)-1)) * sum(u**2 for u in data) - 1 / (len(data) * (len(data)-1)) * sum(data)**2)
    
    # the standard deviation of one time step is o (st_dev over the time frame) * sqrt(t)
    # an estimate of o = s / sqrt(t)
    o = s / np.sqrt(t)
    
    # standard error
    std_err = o / np.sqrt(2 * len(data))
    
    # s = standard deviation of sample
    # o = volatility of the sample
    return (s, o, std_err)


def bsm_calc(S0, K, r, T, vol, div, otype="C"):
    # N = cumulative probability distribution of a variable with a standrard normal distribution, .i.e probability that a variable will be less than 'x'
    # div = continuous dividend yield, can also use for foreign currency risk free rate for ccy options
    if otype == "C":
        Nd1 = ss.norm.cdf(d1(S0, K, r, T, vol, div))
        Nd2 = ss.norm.cdf(d2(S0, K, r, T, vol, div))
        return (S0 * np.exp(-div * T) * Nd1) - (K * np.exp(-r * T) * Nd2)
    else:
        Nd1 = ss.norm.cdf(-d1(S0, K, r, T, vol, div))
        Nd2 = ss.norm.cdf(-d2(S0, K, r, T, vol, div))
        return (K * np.exp(-r * T) * Nd2) - (S0 * np.exp(-div * T) * Nd1)


def d1(S0, K, r, T, vol, div):
    # N(d1) interpretation => the expected stock price at time T in a risk neutral world where S < K counts as 0
    num = (np.log(S0/K) + ((r - div + 0.5 * vol**2) * T)) 
    den = (vol * np.sqrt(T))
    return num / den

def d2(S0, K, r, T, vol, div):
    # N(d2) interpretation => probability a call option will be exercised in a risk neutral world
    return d1(S0, K, r, T, vol, div) - (vol * np.sqrt(T))


def vol_bsm_calc(S0, K, r, T, div, prem, otype="C", imp_vol_guess=0.2, it=100):
    ''' Calculates implied volatility when given the price of a European call option in BS
    
    Parameters
    ==========
    imp_vol_guess : float
        estimate of implied volatility
    it : int 
        number of iterations of process
        
    Returns
    =======
    imp_vol_guess : float
        estimated value for implied vol
    '''
    for i in range(it):
        # reusing code from the price calc to use our vol guess
        vol = imp_vol_guess
        imp_vol_guess -= ((bsm_calc(S0, K, r, T, vol, div, otype) - prem) / calcVega(S0, K, r, T, div, vol_guess=imp_vol_guess))
    return imp_vol_guess


def calcVega(S0, K, r, T, div, vol_guess=None):
    ''' Calculates the change in premium for the option per change in volatility (partial derivative)
    
    Parameters
    =========
    vol_guess : float
        guess at the implied vol needed when calcing the vol from price
    
    Return
    ======
    float
        change in premium per change in volatility
    '''
    
    if vol_guess:
        vol = vol_guess 
    dfq = e ** (div * T)
    return S0 * np.sqrt(T) * dfq * prob_dens_func(d2(S0, K, r, T, vol, div))


def prob_dens_func(x):
    return (1 / np.sqrt(pi*2)) * e ** (((-1) * x**2)/2)


if __name__ == '__main__':
    # print(lognormal_mean(40, 0.16, 0.2, 0.5))
    # print(lognormal_exp_ret_var(20, 0.2, 0.4, 1))
    # print(distribution_rate_of_ret(0.17, 0.20, 3))
    # data = get_daily_return(pd.read_csv('aapl.csv')['Close'])
    # print(estimating_vol(data, t=1/252))
    print(bsm_calc(42, 40, 0.1, 0.5, 0.2, 0.0, "P"))
    print(vol_bsm_calc(21, 20, 0.1, 0.25, 0, 1.875, "C"))
    