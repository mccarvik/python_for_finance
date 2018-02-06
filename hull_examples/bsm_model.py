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


if __name__ == '__main__':
    # print(lognormal_mean(40, 0.16, 0.2, 0.5))
    # print(lognormal_exp_ret_var(20, 0.2, 0.4, 1))
    # print(distribution_rate_of_ret(0.17, 0.20, 3))
    data = get_daily_return(pd.read_csv('aapl.csv')['Close'])
    print(estimating_vol(data, t=1/252))
    