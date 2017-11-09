import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import datetime as dt

from dx import *

PATH = '/home/ubuntu/workspace/python_for_finance/png/scrap/'

def risk_neutral_discounting():
    dates = [dt.datetime(2015, 1, 1), dt.datetime(2015, 7, 1), dt.datetime(2016, 1, 1)]
    
    # getting discount factors assuming a constant rate
    deltas = [0.0, 0.5, 1.0]
    csr = constant_short_rate('csr', 0.05)
    deltas = get_year_deltas(dates)
    print("Discount factors / forward rates for a constant rate")
    print("deltas: " + str(deltas))
    print("discount_factors: " + str(csr.get_discount_factors(dates)[1]))
    print("forward rates: " + str(csr.get_forward_rates(dates)[1]))
    
    # getting
    yields = [0.0025, 0.01, 0.015, 0.025]
    dates = [dt.datetime(2015, 1, 1), dt.datetime(2016, 1, 1), dt.datetime(2020, 1, 1), dt.datetime(2025, 1, 1)]
    dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))
    test_dts = [3, 7, 10]
    deltas = get_year_deltas(dates)
    print("Discount factors / forward rates for a deterministic rate")
    print("aka rate defined from a curve")
    print("deltas: " + str(deltas))
    print("time / yield pairs for curve: " + str(list(zip(deltas, yields))))
    print("discount_factors: " + str(dsr.get_discount_factors(test_dts, dtobjects=False)))
    print("forward rates: " + str(dsr.get_forward_rates(test_dts, dtobjects=False)[1]))
    print("interpolated yields: " + str(dsr.get_interpolated_yields(test_dts, dtobjects=False)))
    
    
def creating_market_environment():
    yields = [0.0025, 0.01, 0.015, 0.025]
    dates = [dt.datetime(2015, 1, 1), dt.datetime(2016, 1, 1), dt.datetime(2020, 1, 1), dt.datetime(2025, 1, 1)]
    dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))
    
    me_1 = market_environment('me_1', dt.datetime(2015,1,1))
    me_1.add_list('symbols', ['AAPL', 'MSFT', 'FB'])
    print(me_1.get_list('symbols'))
    
    me_2 = market_environment('me_2', dt.datetime(2015,1,1))
    me_2.add_constant('volatility', 0.2)
    me_2.add_curve('10_yr', dsr)
    print(me_2.get_curve('10_yr'))
    
    pdb.set_trace()
    me_1.add_environment(me_2)
    print(me_1.get_curve('10_yr'))
    print(me_1.constants)
    print(me_1.lists)
    print(me_1.curves)
    print(me_1.get_curve('10_yr').yield_list[1])


def standard_normal_random_numbers():
    snrn = sn_random_numbers((2, 2, 2), antithetic=False, moment_matching=False, fixed_seed=True)
    print(snrn)
    snrn_mm = sn_random_numbers((2, 3, 2), antithetic=False, moment_matching=True, fixed_seed=True)
    print(snrn_mm)
    print(snrn_mm.mean())
    print(snrn_mm.std())


def geometric_brownian_motion_and_jump_diffusion():
    yields = [0.0025, 0.01, 0.015, 0.025]
    dates = [dt.datetime(2015, 1, 1), dt.datetime(2016, 1, 1), dt.datetime(2020, 1, 1), dt.datetime(2025, 1, 1)]
    dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))
    
    me_gbm = market_environment('me_gbm', dt.datetime(2015, 1, 1))
    me_gbm.add_constant('initial_value', 36.)
    me_gbm.add_constant('volatility', 0.2)
    me_gbm.add_constant('final_date', dt.datetime(2015, 12, 31))
    me_gbm.add_constant('currency', 'EUR')
    # monthly frequency (respcective month end)
    me_gbm.add_constant('frequency', 'M')
    me_gbm.add_constant('paths', 100)
    me_gbm.add_curve('discount_curve', dsr)
    
    gbm = geometric_brownian_motion('gbm', me_gbm)
    gbm.generate_time_grid()
    # print(gbm.time_grid)
    paths_1 = gbm.get_instrument_values()
    pdf = pd.DataFrame(paths_1, index=gbm.time_grid)
    pdf.ix[:, :10].plot(legend=False, figsize=(10, 6))
    plt.savefig(PATH + 'gbm.png', dpi=300)
    print(pdf)


if __name__ == '__main__':
    # risk_neutral_discounting()
    # creating_market_environment()
    # standard_normal_random_numbers()
    geometric_brownian_motion_and_jump_diffusion()