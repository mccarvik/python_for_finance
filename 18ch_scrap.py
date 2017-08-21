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

def derivatives_sim():
    me_gbm = market_environment('me_gbm', dt.datetime(2015, 1, 1))
    me_gbm.add_constant('initial_value', 36.)
    me_gbm.add_constant('volatility', 0.2)
    me_gbm.add_constant('currency', 'EUR')
    me_gbm.add_constant('model', 'gbm')
    me_am_put = market_environment('me_am_put', dt.datetime(2015, 1, 1))
    me_am_put.add_constant('maturity', dt.datetime(2015, 12, 31))
    me_am_put.add_constant('strike', 40.)
    me_am_put.add_constant('currency', 'EUR')
    payoff_func = 'np.maximum(strike - instrument_values, 0)'
    am_put_pos = derivatives_position(
             name='am_put_pos',
             quantity=3,
             underlyings=['gbm'],
             mar_env=me_am_put,
             otype='American single',
             payoff_func=payoff_func)
    am_put_pos.get_info()
    me_jd = market_environment('me_jd', me_gbm.pricing_date)
    
    # add jump diffusion specific parameters
    me_jd.add_constant('lambda', 0.3)
    me_jd.add_constant('mu', -0.75)
    me_jd.add_constant('delta', 0.1)
    
    # add other parameters from gbm
    me_jd.add_environment(me_gbm)

    # needed for portfolio valuation
    me_jd.add_constant('model', 'jd')
    me_eur_call = market_environment('me_eur_call', me_jd.pricing_date)
    me_eur_call.add_constant('maturity', dt.datetime(2015, 6, 30))
    me_eur_call.add_constant('strike', 38.)
    me_eur_call.add_constant('currency', 'EUR')
    payoff_func = 'np.maximum(maturity_value - strike, 0)'
    eur_call_pos = derivatives_position(
             name='eur_call_pos',
             quantity=5,
             underlyings=['jd'],
             mar_env=me_eur_call,
             otype='European single',
             payoff_func=payoff_func)
    underlyings = {'gbm': me_gbm, 'jd' : me_jd}
    positions = {'am_put_pos' : am_put_pos, 'eur_call_pos' : eur_call_pos}

    # discounting object for the valuation
    csr = constant_short_rate('csr', 0.06)
    val_env = market_environment('general', me_gbm.pricing_date)
    val_env.add_constant('frequency', 'W')
    
    # monthly frequency
    val_env.add_constant('paths', 2500)
    val_env.add_constant('starting_date', val_env.pricing_date)
    val_env.add_constant('final_date', val_env.pricing_date)
    
    # not yet known; take pricing_date temporarily
    val_env.add_curve('discount_curve', csr)
    # select single discount_curve for whole portfolio
    pdb.set_trace()
    portfolio = derivatives_portfolio(
                name='portfolio',
                positions=positions,
                val_env=val_env,
                risk_factors=underlyings,
                fixed_seed=False)
    portfolio.get_statistics(fixed_seed=False)
    portfolio.get_statistics(fixed_seed=False)[['pos_value', 'pos_delta', 'pos_vega']].sum()
    
    # aggregate over all positions
    # portfolio.get_positions()
    
    print(portfolio.valuation_objects['am_put_pos'].present_value())
    print(portfolio.valuation_objects['eur_call_pos'].delta())
    path_no = 777
    path_gbm = portfolio.underlying_objects['gbm'].get_instrument_values()[:, path_no]
    path_jd = portfolio.underlying_objects['jd'].get_instrument_values()[:, path_no]

    plt.figure(figsize=(7, 4))
    plt.plot(portfolio.time_grid, path_gbm, 'r', label='gbm')
    plt.plot(portfolio.time_grid, path_jd, 'b', label='jd')
    plt.xticks(rotation=30)
    plt.legend(loc=0); plt.grid(True)
    plt.savefig(PATH + 'dx_port.png', dpi=300)
    plt.close()

    correlations = [['gbm', 'jd', 0.9]]
    port_corr = derivatives_portfolio(
                name='portfolio',
                positions=positions,
                val_env=val_env,
                risk_factors=underlyings,
                correlations=correlations,
                fixed_seed=True)
    print(port_corr.get_statistics())
    path_gbm = port_corr.underlying_objects['gbm'].\
                get_instrument_values()[:, path_no]
    path_jd = port_corr.underlying_objects['jd'].\
                get_instrument_values()[:, path_no]

    plt.figure(figsize=(7, 4))
    plt.plot(portfolio.time_grid, path_gbm, 'r', label='gbm')
    plt.plot(portfolio.time_grid, path_jd, 'b', label='jd')
    plt.xticks(rotation=30)
    plt.legend(loc=0); plt.grid(True)
    plt.savefig(PATH + 'dx_port2.png', dpi=300)
    plt.close()

    pv1 = 5 * port_corr.valuation_objects['eur_call_pos'].\
                present_value(full=True)[1]
    print(pv1)
    pv2 = 3 * port_corr.valuation_objects['am_put_pos'].\
                present_value(full=True)[1]
    print(pv2)
    plt.hist([pv1, pv2], bins=25,
            label=['European call', 'American put']);
    plt.axvline(pv1.mean(), color='r', ls='dashed',
            lw=1.5, label='call mean = %4.2f' % pv1.mean())
    plt.axvline(pv2.mean(), color='r', ls='dotted',
            lw=1.5, label='put mean = %4.2f' % pv2.mean())
    plt.xlim(0, 80); plt.ylim(0, 10000)
    plt.legend()
    plt.savefig(PATH + 'dx_port3.png', dpi=300)
    plt.close()

    pvs = pv1 + pv2
    plt.hist(pvs, bins=50, label='portfolio');
    plt.axvline(pvs.mean(), color='r', ls='dashed',
            lw=1.5, label='mean = %4.2f' % pvs.mean())
    plt.xlim(0, 80); plt.ylim(0, 7000)
    plt.legend()
    plt.savefig(PATH + 'dx_port4.png', dpi=300)
    plt.close()

    # portfolio with correlation
    print(pvs.std())
    # portfolio without correlation
    pv1 = 5 * portfolio.valuation_objects['eur_call_pos'].\
                present_value(full=True)[1]
    pv2 = 3 * portfolio.valuation_objects['am_put_pos'].\
                present_value(full=True)[1]
    print((pv1 + pv2).std())

if __name__ == '__main__': 
    derivatives_sim()