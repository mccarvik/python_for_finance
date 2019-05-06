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


yields = [0.01, 0.05, 0.04, 0.01]
dates = [dt.datetime(2015,1,1), dt.datetime(2015,4,1), dt.datetime(2015,10,1), dt.datetime(2017, 1, 1)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))


##########################################################################################
# Price = at initiation of contract, Valuation = evaluation of price over life of contract
##########################################################################################

def forward_price_calc_cont_disc(price, settle_dt, mat_dt, income_pv=0, yld=0):
    # income_pv is the present value of income from the underlying asset
    t = get_year_deltas([settle_dt, mat_dt])[-1]
    r = dsr.get_interpolated_yields([settle_dt, mat_dt])[1][1]
    if not yld:
        return (price - income_pv) * np.exp(r * t)
    else:
        return (price) * np.exp((r - yld) * t)


def forward_valuation_calc_cont_disc(underlying_px, delivery_px, settle_dt, mat_dt, income_pv=0, yld=0):
    t = get_year_deltas([settle_dt, mat_dt])[-1]
    r = dsr.get_interpolated_yields([settle_dt, mat_dt])[1][1]
    if not yld:
        return underlying_px - income_pv - (delivery_px * np.exp((-1) * r * t))
    else:
        return underlying_px * np.exp((-1) * yld * t) - income_pv - delivery_px * np.exp((-1) * r * t)


def forward_price_calc_stock_idx(price, settle_dt, mat_dt, income_pv=0, div_yld=0):
    # same as forward price calc, just sub dividend yld into calc
    return forward_price_calc_cont_disc(price, settle_dt, mat_dt, yld=div_yld)

    
def forward_price_calc_ccy(price, settle_dt, mat_dt, income_pv=0, foreign_r=0):
    # same as forward price calc, just sub foreign rate into calc
    return forward_price_calc_cont_disc(price, settle_dt, mat_dt, yld=foreign_r)


def forward_price_calc_commodities(price, settle_dt, mat_dt, storage_costs=0, conv_yld=0):
    # same as forward price calc, just sub in convenience yld and storage costs into calc
    return forward_price_calc_cont_disc(price, settle_dt, mat_dt, yld=(conv_yld - storage_costs))


if __name__ == '__main__':
    # print(forward_price_calc_cont_disc(900, dt.datetime(2015,1,1), dt.datetime(2015,10,1), income_pv=39.60))
    # print(forward_price_calc_cont_disc(25, dt.datetime(2015,1,1), dt.datetime(2015,7,1), yld=0.0396))
    # print(forward_valuation_calc_cont_disc(26.28, 24, dt.datetime(2015,1,1), dt.datetime(2015,7,1)))
    
    print(forward_price_calc_stock_idx(1300, dt.datetime(2015,1,1), dt.datetime(2015,4,1), div_yld=0.01))
    print(forward_price_calc_ccy(0.98, dt.datetime(2015,1,1), dt.datetime(2017,1,1), foreign_r=0.03))
    print(forward_price_calc_commodities(120, dt.datetime(2015,1,1), dt.datetime(2015,4,1), conv_yld=0.01, storage_costs=0.03))
    