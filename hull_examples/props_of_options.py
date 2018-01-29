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


def euro_call_bounds(stock_px, strike, settle_dt, mat_dt, r):
    delta_t = get_year_deltas([settle_dt, mat_dt])[-1]
    mx = stock_px
    mn = max(stock_px - strike * np.exp((-1) * r * delta_t), 0)
    return (mx, mn)


def euro_put_bounds(stock_px, strike, settle_dt, mat_dt, r):
    delta_t = get_year_deltas([settle_dt, mat_dt])[-1]
    mx = strike * np.exp((-1) * r * delta_t)
    mn = max(strike * np.exp((-1) * r * delta_t) - stock_px, 0)
    return (mx, mn)
    

def amer_call_bounds(stock_px, strike, settle_dt, mat_dt, r):
    delta_t = get_year_deltas([settle_dt, mat_dt])[-1]
    mx = max(stock_px - (strike / (1+r)**delta_t, 0))
    mn = 0
    return (mx, mn)
    

def amer_put_bounds(stock_px, strike, settle_dt, mat_dt, r):
    mx = max(strike - stock_px, 0)
    mn = 0
    return (mx, mn)


def put_call_parity():
    pass


if __name__ == '__main__':
    print(euro_call_bounds(20, 18, dt.datetime(2015,1,1), dt.datetime(2016,1,1), 0.1))
    print(euro_put_bounds(37, 40, dt.datetime(2015,1,1), dt.datetime(2015,7,1), 0.05))