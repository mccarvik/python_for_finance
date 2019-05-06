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


yields = [0.01, 0.048, 0.04, 0.01]
dates = [dt.datetime(2015,1,1), dt.datetime(2016,1,5), dt.datetime(2016,10,1), dt.datetime(2017, 1, 1)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))


def eurodollar_futures_price(quote):
    # 25$ per basis point move
    return 10000 * (100 - 0.25 *(100 - quote))


# Used to account for the difference between the forward and futures rate
def convexity_adjustment(fut_rat, sigma, settle_dt, T1, T2):
    # sigma = st_dev of short term interest rate in 1 year
    T1 = get_year_deltas([settle_dt, T1])[-1]
    T2 = get_year_deltas([settle_dt, T2])[-1]
    return fut_rat - (0.5) * sigma**2 * T1 * T2


def next_bootstrap_rate(prev_fwd_rt, settle_dt, last_fwd_dt, next_fwd_dt):
    num = prev_fwd_rt * get_year_deltas([last_fwd_dt, next_fwd_dt])[-1] + \
            dsr.get_interpolated_yields([settle_dt, last_fwd_dt])[-1][1] * get_year_deltas([settle_dt, last_fwd_dt])[-1]
    den = get_year_deltas([settle_dt, next_fwd_dt])[-1]
    return num / den
    

# Number of contracts required to hedge against an uncertain change in yld
def duration_based_hedging(fwd_px, dur_port, dur_fut, cont_px):
    return (fwd_px * dur_port) / (cont_px * dur_fut)
    


if __name__ == "__main__":
    # print(eurodollar_futures_price(99.725))
    # print(convexity_adjustment(0.06038, 0.012, dt.datetime(2015,1,1), dt.datetime(2023,1,1), dt.datetime(2023,4,1)))
    # print(next_bootstrap_rate(0.053, dt.datetime(2015,1,1), dt.datetime(2016,2,5), dt.datetime(2016,5,6)))
    print(duration_based_hedging(10000000, 6.8, 9.2, 93062.50))