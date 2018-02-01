import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

PATH = '/home/ubuntu/workspace/python_for_finance/png/hull/wiener_processes/'


def simulate_wiener_process():
    #################################################
    # Markov Property - stochastic process where only the current variable is relevant for the future
    # Wiener Process - Markov as each value is independent of the next + chg over short period of time = sqrt(t)
    # i.e. the random process has a mean change of 0 and a variance of 1 (over some time interval, and vol = sqrt(T))
    # Wiener process pretty much the same thing as Brownian Motion
    # Ito Process - same as Wiener process except the drift rate and the variance rate (csr and vol below) are functions    
    #               dependent on time t and last value x instead of constants
    #################################################
    
    # using this constant rate as a drift factor for the geometric brownan motion
    # csr = constant_short_rate('csr', 0.)
    csr = constant_short_rate('csr', 1.3)
    me_wp = market_environment('me_wp', dt.datetime(2015, 1, 1))
    me_wp.add_constant('initial_value', 1)
    # volatility is the square root of variance
    me_wp.add_constant('volatility', 1.5)
    me_wp.add_constant('final_date', dt.datetime(2015, 12, 31))
    me_wp.add_constant('frequency', 'D')
    me_wp.add_constant('currency', 'EUR')
    me_wp.add_constant('paths', 1)
    me_wp.add_curve('discount_curve', csr)
    
    gbm = geometric_brownian_motion('gbm', me_wp)
    gbm.generate_time_grid()
    paths_1 = gbm.get_instrument_values(fixed_seed=False)
    pdf = pd.DataFrame(paths_1, index=gbm.time_grid)
    pdf.ix[:, :10].plot(legend=False, figsize=(10, 6))
    plt.savefig(PATH + 'wiener_w_drift.png', dpi=300)
    plt.close()
    
    # no drift
    csr = constant_short_rate('csr', 0.0)
    me_wp.add_curve('discount_curve', csr)
    gbm = geometric_brownian_motion('gbm', me_wp)
    gbm.generate_time_grid()
    paths_1 = gbm.get_instrument_values(fixed_seed=False)
    pdf = pd.DataFrame(paths_1, index=gbm.time_grid)
    pdf.ix[:, :10].plot(legend=False, figsize=(10, 6))
    plt.savefig(PATH + 'wiener_wo_drift.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    simulate_wiener_process()