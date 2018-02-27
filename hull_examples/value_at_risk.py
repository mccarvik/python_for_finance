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


def VaR(conf_int, vol, notl, T):
    # vol = 1 year vol
    # T = number of days of VaR
    n_day_vol = vol / sqrt(252) * sqrt(T)
    stdevs_away = ss.norm.ppf(conf_int)
    return stdevs_away * n_day_vol * notl


def VaR_port(conf_int, T, corr_mat, opts):
    # calculate VaR for a portfolio of options, a lot of assumptions for these equations but good exercise
    # vol or o --> o^2 = sum(ai^2 * oi^2) for all i + 2 * corrij * ai * aj * oi * oj for all ij combinations, a = amount invested
    pdb.set_trace()
    # assume a daily vol
    front = sum([(o.px * o.delta * o.vol)**2 for o in opts])
    back = 0
    for i in range(len(opts)):
        for j in range(len(opts)):
            if j >= i:
                continue
            back += opts[i].px * opts[i].delta * opts[j].px * opts[j].delta * opts[i].vol * opts[j].vol * corr_mat[i][j]
    back = 2 * back
    vol = sqrt(front + back)
    return ss.norm.ppf(conf_int) * sqrt(T) * vol


def delta_P(opts, gamma_adj=False):
    # change in portfolio value assuming stdev is actual 1 day return
    delta_P = 0
    for o in opts:
        delta_P += o.px * o.delta * o.vol
    if not gamma_adj:
        return delta_P
    else:
        ga = 0
        for i in range(len(opts)):
            # gamma is the second derivative of the change in port value from changes in option prices
            ga += 0.5 * opts[i].px * opts[i].vol**2 * opts[i].gamma
        return delta_P + ga


class option_sim():
    def __init__(self, px, delta_pos, vol, gamma):
        # delta_pos = delta of the whole position (for the example we werent given the number of contracts)
        self.px = px
        self.delta = delta_pos
        self.vol = vol      # daily vol
        self.gamma = gamma  # gamma
        


if __name__ == '__main__':
    # print(VaR(0.99, 0.3175, 10000000, 10))
    msft = option_sim(120, 1000, 0.02, 0.005)
    att = option_sim(30, 20000, 0.01, 0.002)
    corr_matrix = [['', 0.3], [0.3, '']]
    # print(VaR_port(.95, 5, corr_matrix, [msft, att]))
    print(delta_P([msft, att], False))
    print(delta_P([msft, att], True))