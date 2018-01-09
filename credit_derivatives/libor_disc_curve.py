import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

# import numpy as np
# import pandas as pd
import datetime as dt

from dx import *

# yields = [0.0025, 0.01, 0.015, 0.025]
# dates = [dt.datetime(2015, 1, 1), dt.datetime(2015, 7, 1), dt.datetime(2015, 12, 31), dt.datetime(2018, 12, 31)]
yields = [0.1, 0.15, 0.2, 0.25, 0.3]
dates = [dt.datetime(2016,1,1), dt.datetime(2016,7,1), dt.datetime(2017,1,1), dt.datetime(2017,7,1), dt.datetime(2018,1,1)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))

# returns the value of a money market deposit
def money_market_deposit(settle_dt, mat_dt):
    delta_t = get_year_deltas([settle_dt, mat_dt])[-1]
    yld = dsr.get_interpolated_yields([settle_dt, mat_dt])[-1][1]
    V = 1 + delta_t * yld
    return V


# calculates the discount factor on a given date with given yield curve
def discount_factor(settle_dt, mat_dt):
    # assume trade_dt = settle_dt
    return 1 / money_market_deposit(settle_dt, mat_dt)
    

# calculates the payoff of an FRA at maturity
def fra_payoff(orig_rate, fwd_dt, mat_dt):
    delta_t = get_year_deltas([fwd_dt, mat_dt])
    yld = dsr.get_interpolated_yields([fwd_dt, mat_dt])[-1][1]
    num = delta_t[-1] * (yld - orig_rate)
    den = 1 + (delta_t[-1] * yld)
    return num / den


# calculates the initial rate of an FRA
def fra_rate_calc(settle_dt, fwd_dt, mat_dt):
    num = 1 + get_year_deltas([settle_dt, mat_dt])[-1] * dsr.get_interpolated_yields([settle_dt, mat_dt])[-1][1]
    den = 1 + get_year_deltas([settle_dt, fwd_dt])[-1] * dsr.get_interpolated_yields([settle_dt, fwd_dt])[-1][1]
    return ((num / den) - 1) * (1 / get_year_deltas([fwd_dt, mat_dt])[-1])
    

# calculates the price of an interest rate futures contract given settle_dt
def ir_futures_calc(settle_dt, fwd_dt, mat_dt):
    # TODO - at time settle_dt get the rate for fwd_dt at tenor to mat_dt
    # Forward = [(1 + spot rate for year x)x/ (1 + spot rate for year y)y] - 1
    # print(((1 + 0.3)**2 / (1 + 0.2)**1) - 1)
    # TODO - probably something wrong here, need to use fram.py get_forward_rate function
    ylds = dsr.get_interpolated_yields([settle_dt, fwd_dt, mat_dt])
    fwd_rate = ((1 + ylds[-1][1])**get_year_deltas([settle_dt, mat_dt])[-1]) / ((1 + ylds[-2][1])**get_year_deltas([settle_dt, fwd_dt])[-1]) - 1
    return 100 - fwd_rate * 100


if __name__ == '__main__':
    # print(money_market_deposit(dt.datetime(2015, 1, 1), dt.datetime(2015, 5, 1)))
    # print(discount_factor(dt.datetime(2015, 1, 1), dt.datetime(2015, 5, 1)))
    # print(fra_payoff(0.055, dt.datetime(2016,1,1), dt.datetime(2016,7,1)))
    # print(fra_rate_calc(dt.datetime(2016,1,1), dt.datetime(2016,4,1), dt.datetime(2016,10,1)))
    print(ir_futures_calc(dt.datetime(2016,1,1), dt.datetime(2017,1,1), dt.datetime(2018,1,1)))
    