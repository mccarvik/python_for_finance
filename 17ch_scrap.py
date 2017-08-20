import sys, pdb
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import datetime as dt

from dx import *

PATH = '/home/ubuntu/workspace/python_for_finance/png/dx/'

def val_european_monte_carlo():
    me_gbm = market_environment('me_gbm', dt.datetime(2015, 1, 1))
    me_gbm.add_constant('initial_value', 36.)
    me_gbm.add_constant('volatility', 0.2)
    me_gbm.add_constant('final_date', dt.datetime(2015, 12, 31))
    me_gbm.add_constant('currency', 'EUR')
    me_gbm.add_constant('frequency', 'M')
    me_gbm.add_constant('paths', 10000)
    csr = constant_short_rate('csr', 0.06)
    me_gbm.add_curve('discount_curve', csr)
    gbm = geometric_brownian_motion('gbm', me_gbm)
    me_call = market_environment('me_call', me_gbm.pricing_date)
    me_call.add_constant('strike', 40.)
    me_call.add_constant('maturity', dt.datetime(2015, 12, 31))
    me_call.add_constant('currency', 'EUR')
    payoff_func = 'np.maximum(maturity_value - strike, 0)'
    eur_call = valuation_mcs_european_single('eur_call', underlying=gbm,
                        mar_env=me_call, payoff_func=payoff_func)
    print(eur_call.present_value())
    print(eur_call.delta())
    print(eur_call.vega())
    s_list = np.arange(34., 46.1, 2.)
    p_list = []; d_list = []; v_list = []
    for s in s_list:
        eur_call.update(initial_value=s)
        p_list.append(eur_call.present_value(fixed_seed=True))
        d_list.append(eur_call.delta())
        v_list.append(eur_call.vega())

    
    plot_option_stats("eur_call", s_list, p_list, d_list, v_list)

    payoff_func = 'np.maximum(0.33 * (maturity_value + max_value) - 40, 0)'
    # payoff dependent on both the simulated maturity value
    # and the maximum value
    eur_as_call = valuation_mcs_european_single('eur_as_call', underlying=gbm,
                            mar_env=me_call, payoff_func=payoff_func)
    s_list = np.arange(34., 46.1, 2.)
    p_list = []; d_list = []; v_list = []
    for s in s_list:
        eur_as_call.update(s)
        p_list.append(eur_as_call.present_value(fixed_seed=True))
        d_list.append(eur_as_call.delta())
        v_list.append(eur_as_call.vega())
    plot_option_stats("eur_asian_call", s_list, p_list, d_list, v_list)

def val_american_monte_carlo():
    me_gbm = market_environment('me_gbm', dt.datetime(2015, 1, 1))
    me_gbm.add_constant('initial_value', 36.)
    me_gbm.add_constant('volatility', 0.2)
    me_gbm.add_constant('final_date', dt.datetime(2016, 12, 31))
    me_gbm.add_constant('currency', 'EUR')
    me_gbm.add_constant('frequency', 'W')
    # weekly frequency
    me_gbm.add_constant('paths', 5000)
    csr = constant_short_rate('csr', 0.06)
    me_gbm.add_curve('discount_curve', csr)
    gbm = geometric_brownian_motion('gbm', me_gbm)
    payoff_func = 'np.maximum(strike - instrument_values, 0)'
    me_am_put = market_environment('me_am_put', dt.datetime(2015, 1, 1))
    me_am_put.add_constant('maturity', dt.datetime(2015, 12, 31))
    me_am_put.add_constant('strike', 40.)
    me_am_put.add_constant('currency', 'EUR')
    am_put = valuation_mcs_american_single('am_put', underlying=gbm,
                    mar_env=me_am_put, payoff_func=payoff_func)
    print(am_put.present_value(fixed_seed=True, bf=5))
    
    ls_table = []
    for initial_value in (36., 38., 40., 42., 44.): 
        for volatility in (0.2, 0.4):
            for maturity in (dt.datetime(2015, 12, 31),
                             dt.datetime(2016, 12, 31)):
                am_put.update(initial_value=initial_value,
                                volatility=volatility,
                                maturity=maturity)
                ls_table.append([initial_value,
                                volatility,
                                maturity,
                                am_put.present_value(bf=5)])
    print("S0  | Vola | T | Value")
    print(22 * "-")
    for r in ls_table:
        print("%d  | %3.1f  | %d | %5.3f" % (r[0], r[1], r[2].year - 2014, r[3]))
    am_put.update(initial_value=36.)
    print(am_put.delta())
    print(am_put.vega())

if __name__ == '__main__':
    # val_european_monte_carlo()
    val_american_monte_carlo()