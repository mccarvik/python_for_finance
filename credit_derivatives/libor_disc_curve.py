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
from utils.utils import *

from dx import *

# yields = [0.0025, 0.01, 0.015, 0.025]
# dates = [dt.datetime(2015, 1, 1), dt.datetime(2015, 7, 1), dt.datetime(2015, 12, 31), dt.datetime(2018, 12, 31)]
yields = [0.01, 0.034, 0.071, 00.037, 0.074]
dates = [dt.datetime(2015,1,1), dt.datetime(2015,4,1), dt.datetime(2015,7,1), dt.datetime(2015,10,1), dt.datetime(2016,1,1)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))

# returns the value of a money market deposit
def money_market_deposit(settle_dt, mat_dt):
    delta_t = get_year_deltas([settle_dt, mat_dt])[-1]
    # Need to add the first date of the curve to use the library, shouldn't affect anything as we are only taking the last result anyway
    yld = dsr.get_interpolated_yields([dsr.yield_list[0][0], settle_dt, mat_dt])[-1][1]
    V = 1 + delta_t * yld
    return V


# calculates the discount factor on a given date with given yield curve
def discount_factor(settle_dt, mat_dt):
    # assume trade_dt = settle_dt
    return 1 / money_market_deposit(settle_dt, mat_dt)
    

# calculates the payoff of an FRA at maturity
def fra_payoff(orig_rate, settle_dt, mat_dt):
    delta_t = get_year_deltas([settle_dt, mat_dt])
    yld = dsr.get_interpolated_yields([settle_dt, mat_dt])[-1][1]
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
    ylds = dsr.get_interpolated_yields([settle_dt, fwd_dt, mat_dt])
    fwd_rate = ((1 + ylds[-1][1])**get_year_deltas([settle_dt, mat_dt])[-1]) / ((1 + ylds[-2][1])**get_year_deltas([settle_dt, fwd_dt])[-1]) - 1
    return 100 - fwd_rate * 100
    

def swap_rate_calc(settle_dt, mat_dt, fixed_prds, float_prds):
    PV_last = discount_factor(settle_dt, mat_dt)
    num = 1 - PV_last
    cfs = createCashFlows(settle_dt, 1 / float_prds, mat_dt, 0, 1, par_cf=False)
    PV_sum = sum([discount_factor(settle_dt, d) for d, c in cfs])
    return (num / PV_sum)
    

# return value from perspective of receiver (receives fixed)
def swap_valuation_calc(fix_rate, orig_first_flt_rate, start_dt, settle_dt, mat_dt, fixed_prds, float_prds):
    # PV_fixed
    fix_cfs = createCashFlows(start_dt, 1 / fixed_prds, mat_dt, 0, 1, par_cf=False)
    fix_cfs = [discount_factor(settle_dt, d) for d, c in fix_cfs]
    PV_fixed = fix_cfs[-1] + sum(fix_cfs) * fix_rate
    
    # PV_float
    # just need PV of first payment, rest will have a PV of one
    first_flt_dt = createCashFlows(start_dt, 1 / float_prds, mat_dt, 0, 1, par_cf=False)[0][0]
    PV_float = discount_factor(settle_dt, first_flt_dt) * (1 + orig_first_flt_rate)
    
    return PV_fixed - PV_float


if __name__ == '__main__':
    # print(money_market_deposit(dt.datetime(2015, 1, 1), dt.datetime(2015, 5, 1)))
    # print(discount_factor(dt.datetime(2015, 1, 1), dt.datetime(2015, 5, 1)))
    # print(fra_payoff(0.055, dt.datetime(2016,1,1), dt.datetime(2016,7,1)))
    # print(fra_rate_calc(dt.datetime(2016,1,1), dt.datetime(2016,4,1), dt.datetime(2016,10,1)))
    # print(ir_futures_calc(dt.datetime(2016,1,1), dt.datetime(2017,1,1), dt.datetime(2018,1,1)))
    
    print(swap_rate_calc(dt.datetime(2015,1,1), dt.datetime(2016,1,1), 4, 4))
    print(swap_valuation_calc(0.0392, 0.036, dt.datetime(2015,1,1), dt.datetime(2015,4,1), dt.datetime(2016,1,1), 2, 2))
    