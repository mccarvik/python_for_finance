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

no_paths = 10000
    
# Market Environment setup
me = market_environment('me', dt.datetime(2015, 1, 1))
me.add_constant('initial_value', 35)
me.add_constant('volatility', 0.2)
me.add_constant('maturity', dt.datetime(2015, 12, 31))
me.add_constant('final_date', dt.datetime(2016, 1, 31))
me.add_constant('currency', 'EUR')
me.add_constant('frequency', 'W')       # M = monthly, W = weekly, A = Annual
me.add_constant('paths', no_paths)

# Constant Curve vs. Normal Upward sloping Curve
# me.add_curve('discount_curve', dsr)
me.add_curve('discount_curve', r)
gbm = geometric_brownian_motion('gbm', me)

euro_call_payoff = 'np.maximum(maturity_value - strike, 0)'
# payoff = 'np.maximum(np.minimum(maturity_value) * 2 - 50, 0)'


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
    
    # calc greek movement over ttm
    t_list = call_eur.underlying.time_grid
    pv = []; de = []; ve = []; th = []; rh = []; ga = []
    for t in t_list[3::4]:
        call_eur.update(maturity=t)
        pv.append(call_eur.present_value())
        de.append(call_eur.delta(0.5))
        ve.append(call_eur.vega(0.2))
        th.append(call_eur.theta())
        rh.append(call_eur.rho())
        ga.append(call_eur.gamma())

    pdb.set_trace()
    plot_option_stats_full(t_list[3::4], pv, de, ve, th, rh, ga, PATH, "maturity")
    call_eur.update(maturity=dt.datetime(2015, 12, 31))
    
    # calc greek movement over strikes
    k_list = np.arange(26, 46.1, 2)
    pv = []; de = []; ve = []; th = []; rh = []; ga = []
    for k in k_list:
        call_eur.update(strike=k)
        pv.append(call_eur.present_value())
        de.append(call_eur.delta(0.5))
        ve.append(call_eur.vega(0.2))
        th.append(call_eur.theta())
        rh.append(call_eur.rho())
        ga.append(call_eur.gamma())
    
    plot_option_stats_full(k_list, pv, de, ve, th, rh, ga, PATH, "strikes")
    call_eur.update(strike=40.0)
    
    
    # calc greek movement over volatility
    v_list = np.arange(0.05, 0.6, 0.05)
    pv = []; de = []; ve = []; th = []; rh = []; ga = []
    for v in v_list:
        call_eur.update(volatility=v)
        pv.append(call_eur.present_value())
        de.append(call_eur.delta(0.5))
        ve.append(call_eur.vega(0.2))
        th.append(call_eur.theta())
        rh.append(call_eur.rho())
        ga.append(call_eur.gamma())

    plot_option_stats_full(v_list, pv, de, ve, th, rh, ga, PATH, "volatility")
    call_eur.update(volatility=0.2)
    
    
    # calc greek movement over underlying price
    p_list = np.arange(10., 60, 5)
    pv = []; de = []; ve = []; th = []; rh = []; ga = []
    for p in p_list:
        call_eur.update(initial_value=p)
        pv.append(call_eur.present_value())
        de.append(call_eur.delta(0.5))
        ve.append(call_eur.vega(0.2))
        th.append(call_eur.theta())
        rh.append(call_eur.rho())
        ga.append(call_eur.gamma())
    
    plot_option_stats_full(p_list, pv, de, ve, th, rh, ga, PATH, "underlying")


def mcs_american_single():
    me.add_constant('maturity', dt.datetime(2015, 12, 31))
    me.add_constant('strike', 40.)
    put_ame = valuation_mcs_american_single(
                name='put_eur',
                underlying=gbm,
                mar_env=me,
                payoff_func='np.maximum(strike - instrument_values, 0)')
    print(put_ame.present_value())
    print(put_ame.delta())
    print(put_ame.vega())
    
    k_list = np.arange(26., 46.1, 2.)
    pv = []; de = []; ve = []; th = []; rh = []; ga = []
    for k in k_list:
        put_ame.update(strike=k)
        pv.append(put_ame.present_value())
        de.append(put_ame.delta(.5))
        ve.append(put_ame.vega(0.2))
        th.append(put_ame.theta())
        rh.append(put_ame.rho())
        ga.append(put_ame.gamma())
    
    plot_option_stats_full(k_list, pv, de, ve, th, rh, ga, PATH, "amer_put_strikes")


def port_valuation():
    me.add_constant('model', 'gbm')
    put = derivatives_position(
                name='put',  # name of position
                quantity=1,  # number of instruments
                underlyings=['gbm'],  # relevant risk factors
                mar_env=me,  # market environment
                otype='American single',  # the option type
                payoff_func='np.maximum(40. - instrument_values, 0)') # the payoff funtion
    # put.get_info()
    
    # Uncorrelated
    me_jump = market_environment('me_jump', dt.datetime(2015, 1, 1))
    me_jump.add_environment(me)
    me_jump.add_constant('lambda', 0.8)
    me_jump.add_constant('mu', -0.8)
    me_jump.add_constant('delta', 0.1)
    me_jump.add_constant('model', 'jd')

    call_jump = derivatives_position(
                    name='call_jump',
                    quantity=3,
                    underlyings=['jd'],
                    mar_env=me_jump,
                    otype='European single',
                    payoff_func='np.maximum(maturity_value - 36., 0)')
    
    risk_factors = {'gbm': me, 'jd' : me_jump}
    positions = {'put' : put, 'call_jump' : call_jump}

    val_env = market_environment('general', dt.datetime(2015, 1, 1))
    val_env.add_constant('frequency', 'M')
    val_env.add_constant('paths', 10000)
    val_env.add_constant('starting_date', val_env.pricing_date)
    val_env.add_constant('final_date', val_env.pricing_date)
    val_env.add_curve('discount_curve', r)

    port = derivatives_portfolio(
                name='portfolio',  # name 
                positions=positions,  # derivatives positions
                val_env=val_env,  # valuation environment
                risk_factors=risk_factors, # relevant risk factors
                correlations=False,  # correlation between risk factors
                fixed_seed=False,  # fixed seed for randon number generation
                parallel=False)  # parallel valuation of portfolio positions

    # portolio delta, vega, etc. just the sum when uncorrelated
    stats = port.get_statistics()
    print(stats)
    print(stats[['pos_value', 'pos_delta', 'pos_vega']].sum())
    print(port.get_values())
    # print(port.get_positions())
    
    
    # Correlated
    # 90% correlated
    correlations = [['gbm', 'jd', 0.9]]

    port = derivatives_portfolio(
                name='portfolio',
                positions=positions,
                val_env=val_env,
                risk_factors=risk_factors,
                correlations=correlations,
                fixed_seed=True,
                parallel=False)
    
    pdb.set_trace()
    port.get_statistics()
    print(port.val_env.lists['cholesky_matrix'])

    path_no = 0
    paths1 = port.underlying_objects['gbm'].get_instrument_values()[:, path_no]
    paths2 = port.underlying_objects['jd'].get_instrument_values()[:, path_no]
    
    # highly correlated underlyings
    # -- with a large jump for one risk factor
    plt.figure(figsize=(10, 6))
    plt.plot(port.time_grid, paths1, 'r', label='gbm')
    plt.plot(port.time_grid, paths2, 'b', label='jd')
    plt.gcf().autofmt_xdate()
    plt.legend(loc=0); plt.grid(True)
    plt.savefig(PATH + "correlated_port.png", dpi=300)
    plt.close()
    
    
    


if __name__ == '__main__':
    # mcs_european_single()
    # mcs_american_single()
    port_valuation()