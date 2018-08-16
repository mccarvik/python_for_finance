import os, pdb, time, sys, time
sys.path.append("/usr/local/lib/python3.4/dist-packages")
sys.path.append("/home/ubuntu/workspace/python_for_finance/")
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from dawp.chs.ch5 import BSM_call_value

mpl.rcParams['font.family'] = 'serif'

PNG_PATH = '../png/2ch/'


def bsm_plot():
    #
    # European Call Option Value Plot
    # 02_mbv/BSM_value_plot.py
    # Model and Option Parameters
    K = 8000  # strike price
    T = 1.0  # time-to-maturity
    r = 0.025  # constant, risk-less short rate
    vol = 0.2  # constant volatility
    
    # Sample Data Generation
    S = np.linspace(4000, 12000, 150)  # vector of index level values
    h = np.maximum(S - K, 0)  # inner value of option
    C = [BSM_call_value(S0, K, 0, T, r, vol) for S0 in S]
    # calculate call option values
    
    # Graphical Output of intrinsic value vs. time value of option
    plt.figure()
    plt.plot(S, h, 'b-.', lw=2.5, label='inner value')
    # plot inner value at maturity
    plt.plot(S, C, 'r', lw=2.5, label='present value')
    # plot option present value
    plt.grid(True)
    plt.legend(loc=0)
    plt.xlabel('index level $S_0$')
    plt.ylabel('present value $C(t=0)$')
    plt.savefig(PNG_PATH + "BSM_value_plot", dpi=300)
    plt.close()


def inner_value():
    # Option Strike
    K = 8000
    
    # Graphical Output
    S = np.linspace(7000, 9000, 100)  # index level values
    h = np.maximum(S - K, 0)  # inner values of call option
    
    # Hockey stick graphs of option value
    plt.figure()
    plt.plot(S, h, lw=2.5)  # plot inner values at maturity
    plt.xlabel('index level $S_t$ at maturity')
    plt.ylabel('inner value of European call option')
    plt.grid(True)
    plt.savefig(PNG_PATH + "inner_value", dpi=300)
    plt.close()
    
    
if __name__ == '__main__':
    # inner_value()
    bsm_plot()