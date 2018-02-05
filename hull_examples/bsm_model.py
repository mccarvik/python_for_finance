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
# import pandas as pd
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

def lognormal_exp_ret_var(S0, exp_ret, vol, T):
    # lognormal expected return and standard dev
    # E(St) = S0e^(uT)
    # var(St) = S0^2*e^(2uT)*(e^(o^2*T)-1)
    ret = S0 * np.exp(exp_ret * T)
    var = S0**2 * np.exp(2*exp_ret*T) * (np.exp(vol**2 * T) - 1)
    st_dev = np.sqrt(var)
    return (ret, st_dev)
    

if __name__ == '__main__':
    # print(lognormal_mean(40, 0.16, 0.2, 0.5))
    print(lognormal_exp_ret_var(20, 0.2, 0.4, 1))