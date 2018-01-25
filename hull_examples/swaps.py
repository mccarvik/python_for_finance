import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")

from dx import *
from utils.utils import *
from hull_examples.interest_rates import calc_fwd_rate

import numpy as np
# import pandas as pd
import datetime as dt

yields = [0.09, 0.09]
dates = [dt.datetime(2015,1,1), dt.datetime(2019,1,1)]
# yields = [0.01, 0.028, 0.032, 0.034]
# dates = [dt.datetime(2014,12,31), dt.datetime(2015,4,1), dt.datetime(2015,10,1), dt.datetime(2016,4,8)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))


# returns the value of a money market deposit
def money_market_deposit(settle_dt, mat_dt, curve=False):
    if curve:
        crv = curve
    else:
        crv = dsr
    delta_t = get_year_deltas([settle_dt, mat_dt])[-1]
    # Need to add the first date of the curve to use the library, shouldn't affect anything as we are only taking the last result anyway
    yld = crv.get_interpolated_yields([crv.yield_list[0][0], settle_dt, mat_dt])[-1][1]
    V = 1 + delta_t * yld
    return V


# calculates the discount factor on a given date with given yield curve
def discount_factor(settle_dt, mat_dt, curve=False):
    # assume trade_dt = settle_dt
    if not curve:
        return 1 / money_market_deposit(settle_dt, mat_dt)
    else:
        return 1 / money_market_deposit(settle_dt, mat_dt, curve)


# return value from perspective of receiver (receives fixed)
def swap_valuation_calc(fix_rate, orig_first_flt_rate, start_dt, settle_dt, mat_dt, fixed_prds, float_prds):
    # Calcualtes swap value as a difference of a fixed bond and a floating bond based on notional of 100
    
    # PV_fixed
    fix_cfs = createCashFlows(start_dt, 1 / fixed_prds, mat_dt, fix_rate, 100)
    dfs = [discount_factor(settle_dt, d) * c for d, c in fix_cfs]
    PV_fixed = sum(dfs)
    
    # PV_float
    # just need PV of first payment, rest will have a PV of one
    first_flt_dt = createCashFlows(start_dt, 1 / float_prds, mat_dt, 0, 1, par_cf=False)[0][0]
    PV_float = discount_factor(settle_dt, first_flt_dt) * (100 + 100 * orig_first_flt_rate / float_prds)
    
    return PV_fixed - PV_float


# return value from perspective of receiver (receives fixed)
def swap_valuation_calc_fra(fix_rate, orig_first_flt_rate, start_dt, settle_dt, mat_dt, fixed_prds, float_prds):
    # Calcualtes swap value in terms of FRAs and forward rates
    fix_cfs = createCashFlows(start_dt, 1 / fixed_prds, mat_dt, fix_rate, 1, par_cf=False)
    
    # Need to count for the fact that the first floating payment is already established
    flt_cfs = [orig_first_flt_rate / fixed_prds]
    for i in range(len(fix_cfs[1:])):
        flt_cfs.append(calc_fwd_rate(settle_dt, fix_cfs[i][0], fix_cfs[i+1][0], dsr) / fixed_prds)
    
    net_cfs = [fix[1] - flt for fix, flt in zip(fix_cfs, flt_cfs)]
    net_cfs = [(d[0],cf) for d, cf in zip(fix_cfs, net_cfs)]
    net_pv = [discount_factor(settle_dt, d) * cf for d, cf in net_cfs]
    return sum(net_pv)

# returns from the the perspective of receving foreign and paying domestic, fixed-for-fixed
def swap_valuation_calc_ccy(fix_rate_dom, fix_rate_for, notl_dom, notl_for, start_dt, settle_dt, mat_dt, fixed_prds, exch_r, for_curve):
    # Calcualtes swap value as a difference of a fixed bond in the domestic ccy and a fixed bond in the foreign ccy
    
    # PV_fixed
    dom_cfs = createCashFlows(start_dt, 1 / fixed_prds, mat_dt, fix_rate_dom, notl_dom)
    dom_dfs = [discount_factor(settle_dt, d) * c for d, c in dom_cfs]
    PV_dom = sum(dom_dfs)
    
    for_cfs = createCashFlows(start_dt, 1 / fixed_prds, mat_dt, fix_rate_for, notl_for)
    for_dfs = [discount_factor(settle_dt, d, for_curve) * c for d, c in for_cfs]
    PV_for = sum(for_dfs)
    
    return (PV_for / exch_r) - PV_dom
    

if __name__ == '__main__':
    # print(swap_valuation_calc(0.03, 0.029, dt.datetime(2014,10,1), dt.datetime(2014,12,31), dt.datetime(2016,4,6), 2, 2))
    # print(swap_valuation_calc_fra(0.03, 0.029, dt.datetime(2014,10,1), dt.datetime(2014,12,31), dt.datetime(2016,3,31), 2, 2))
    yields = [0.04, 0.04]
    dates = [dt.datetime(2015,1,1), dt.datetime(2019,1,1)]
    foreign = deterministic_short_rate('foreign', list(zip(dates, yields)))
    print(swap_valuation_calc_ccy(0.08, 0.05, 10, 1200, dt.datetime(2015,1,1), dt.datetime(2015,1,1), dt.datetime(2018,1,1), 1, 110, foreign))    
    