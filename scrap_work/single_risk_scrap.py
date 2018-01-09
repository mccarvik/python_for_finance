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

PATH = '/home/ubuntu/workspace/python_for_finance/png/scrap/single_risk/'

colormap='RdYlBu_r'
lw=1.25
figsize=(10, 6)
legend=False


r = constant_short_rate('r', 0.06)
yields = [0.0025, 0.01, 0.015]
dates = [dt.datetime(2015, 1, 1), dt.datetime(2015, 7, 1), dt.datetime(2015, 12, 31)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))

no_paths = 1000
    
# Market Environment setup
me = market_environment('me', dt.datetime(2015, 1, 1))
me.add_constant('initial_value', 36)
me.add_constant('volatility', 0.2)
me.add_constant('final_date', dt.datetime(2015, 12, 31))
me.add_constant('currency', 'EUR')
me.add_constant('frequency', 'W')       # M = monthly, W = weekly, A = Annual
me.add_constant('paths', no_paths)

# Constant Curve vs. Normal Upward sloping Curve
# me.add_curve('discount_curve', dsr)
me.add_curve('discount_curve', r)
gbm = geometric_brownian_motion('gbm', me)

euro_call_payoff = 'np.maximum(maturity_value - strike, 0)'

def mcs_european_single():
    me.add_constant('maturity', dt.datetime(2015, 12, 31))
    me.add_constant('strike', 40.)
    call_eur = valuation_mcs_european_single(
                    name = 'call_eur',
                    underlying=gbm,
                    mar_env=me,
                    payoff_func=euro_call_payoff
                    )
    print(call_eur.present_value())
    print(call_eur.delta())
    print(call_eur.vega())
    print(call_eur.theta())
    print(call_eur.rho())
    print(call_eur.gamma())
    
    k_list = np.arange(26., 46.1, 2)
    pv = []; de = []; ve = []; th = []; rh = []; ga = []
    for k in k_list:
        call_eur.update(strike=k)
        pv.append(call_eur.present_value())
        de.append(call_eur.delta(0.5))
        ve.append(call_eur.vega(0.2))
        th.append(call_eur.theta())
        rh.append(call_eur.rho())
        ga.append(call_eur.gamma())
    
    plot_option_stats_full(k_list, pv, de, ga, ve, th, rh, PATH, "strikes")




if __name__ == '__main__':
    mcs_european_single()