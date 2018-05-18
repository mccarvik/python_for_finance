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

PATH = '/home/ubuntu/workspace/python_for_finance/png/scrap/multi_risk/'

r = constant_short_rate('r', 0.06)
me1 = market_environment('me1', dt.datetime(2015, 1, 1))
me2 = market_environment('me2', dt.datetime(2015, 1, 1))

me1.add_constant('initial_value', 36.)
me1.add_constant('volatility', 0.1)  # low volatility
me1.add_constant('currency', 'EUR')
me1.add_constant('model', 'gbm')
me2.add_environment(me1)
me2.add_constant('initial_value', 36.)
me2.add_constant('volatility', 0.5)  # high volatility

risk_factors = {'gbm1' : me1, 'gbm2' : me2}
correlations = [['gbm1', 'gbm2', 0.5]]

val_env = market_environment('val_env', dt.datetime(2015, 1, 1))
val_env.add_constant('starting_date', val_env.pricing_date)
val_env.add_constant('final_date', dt.datetime(2015, 12, 31))
val_env.add_constant('frequency', 'W')
val_env.add_constant('paths', 5000)
val_env.add_curve('discount_curve', r)
val_env.add_constant('maturity', dt.datetime(2015, 12, 31))
val_env.add_constant('currency', 'EUR')



def valuation_mcs_euro_multi():
    # European maximum call option
    payoff_func = "np.maximum(np.maximum(maturity_value['gbm1'], maturity_value['gbm2']) - 38, 0)"
    
    # Will calculate metrics under two separate evnironemnts: low vol and hi vol
    vc = valuation_mcs_european_multi(
            name='European maximum call',  # name
            val_env=val_env,  # valuation environment
            risk_factors=risk_factors,  # the relevant risk factors
            correlations=correlations,  # correlations between risk factors
            payoff_func=payoff_func)  # payoff function
    
    # Printing out some stats
    # print(vc.risk_factors)
    # print(vc.underlying_objects)
    # print(vc.correlations)
    # print(vc.correlation_matrix)
    # print(vc.val_env.get_list('cholesky_matrix'))
    # print(np.shape(vc.generate_payoff()))
    # print(vc.present_value())
    # vc.update('gbm1', initial_value=50.)
    # print(vc.present_value())
    
    # vc.update('gbm2', volatility=0.6)
    # print(vc.present_value())

    # vc.update('gbm1', initial_value=36., volatility=0.1)
    # vc.update('gbm2', initial_value=36., volatility=0.5)
    # print(vc.delta('gbm2', interval=0.5))
    # print(vc.gamma('gbm2'))
    # print(vc.dollar_gamma('gbm2'))
    # print(vc.vega('gbm1'))
    # print(vc.theta())
    # print(vc.rho(interval=0.1))
    # print(val_env.curves['discount_curve'].short_rate)
    
    s_list = np.arange(28., 46.1, 2.)
    pv = []; de = []; ve = []
    for s in s_list:
        vc.update('gbm1', initial_value=s)
        pv.append(vc.present_value())
        de.append(vc.delta('gbm1', .5))
        ve.append(vc.vega('gbm1', 0.2))
    vc.update('gbm1', initial_value=36.)
    plot_option_stats('multi_risk/risk_factors1', s_list, pv, de, ve)
    
    s_list = np.arange(28., 46.1, 2.)
    pv = []; de = []; ve = []
    for s in s_list:
        vc.update('gbm2', initial_value=s)
        pv.append(vc.present_value())
        de.append(vc.delta('gbm2', .5))
        ve.append(vc.vega('gbm2', 0.2))
    plot_option_stats('multi_risk/risk_factors2', s_list, pv, de, ve)


def negative_correlation():
    correlationscorrelat  = [['gbm1', 'gbm2', -0.9]]
    # European maximum call option
    payoff_func = "np.maximum(np.maximum(maturity_value['gbm1'], maturity_value['gbm2']) - 38, 0)"
    vc = valuation_mcs_european_multi(
                name='European maximum call',
                val_env=val_env,
                risk_factors=risk_factors,
                correlations=correlations,
                payoff_func=payoff_func)

    s_list = np.arange(28., 46.1, 2.)
    pv = []; de = []; ve = []
    for s in s_list:
        vc.update('gbm1', initial_value=s)
        pv.append(vc.present_value())
        de.append(vc.delta('gbm1', .5))
        ve.append(vc.vega('gbm1', 0.2))
    vc.update('gbm1', initial_value=36.)
    plot_option_stats('multi_risk/neg_risk_factors1', s_list, pv, de, ve)
    
    s_list = np.arange(28., 46.1, 2.)
    pv = []; de = []; ve = []
    for s in s_list:
        vc.update('gbm2', initial_value=s)
        pv.append(vc.present_value())
        de.append(vc.delta('gbm2', .5))
        ve.append(vc.vega('gbm2', 0.2))
    plot_option_stats('multi_risk/neg_risk_factors2', s_list, pv, de, ve)


def pos_surface_corr():
    correlations = [['gbm1', 'gbm2', 0.5]]
    # European maximum call option
    payoff_func = "np.maximum(np.maximum(maturity_value['gbm1'], maturity_value['gbm2']) - 38, 0)"
    vc = valuation_mcs_european_multi(
                name='European maximum call',
                val_env=val_env,
                risk_factors=risk_factors,
                correlations=correlations,
                payoff_func=payoff_func)

    pdb.set_trace()
    asset_1 = np.arange(28., 46.1, 4.)  # range of initial values
    asset_2 = asset_1
    a_1, a_2 = np.meshgrid(asset_1, asset_2)
    # two-dimensional grids out of the value vectors
    value = np.zeros_like(a_1)
    
    for i in range(np.shape(value)[0]):
        for j in range(np.shape(value)[1]):
            vc.update('gbm1', initial_value=a_1[i, j])
            vc.update('gbm2', initial_value=a_2[i, j])
            value[i, j] = vc.present_value()
    plot_greeks_3d([a_1, a_2, value], ['gbm1', 'gbm2', 'present value'])
    
    # Delta Surfaces
    delta_1 = np.zeros_like(a_1)
    delta_2 = np.zeros_like(a_1)
    for i in range(np.shape(delta_1)[0]):
        for j in range(np.shape(delta_1)[1]):
            vc.update('gbm1', initial_value=a_1[i, j])
            vc.update('gbm2', initial_value=a_2[i, j])
            delta_1[i, j] = vc.delta('gbm1')
            delta_2[i, j] = vc.delta('gbm2')
    plot_greeks_3d([a_1, a_2, delta_1], ['gbm1', 'gbm2', 'delta gbm1'])
    plot_greeks_3d([a_1, a_2, delta_2], ['gbm1', 'gbm2', 'delta gbm2'])
    
    # Vega Surfaces
    vega_1 = np.zeros_like(a_1)
    vega_2 = np.zeros_like(a_1)
    for i in range(np.shape(vega_1)[0]):
        for j in range(np.shape(vega_1)[1]):
            vc.update('gbm1', initial_value=a_1[i, j])
            vc.update('gbm2', initial_value=a_2[i, j])
            vega_1[i, j] = vc.vega('gbm1')
            vega_2[i, j] = vc.vega('gbm2')
    plot_greeks_3d([a_1, a_2, vega_1], ['gbm1', 'gbm2', 'vega gbm1'])
    plot_greeks_3d([a_1, a_2, vega_2], ['gbm1', 'gbm2', 'vega gbm2'])

    # restore initial values
    vc.update('gbm1', initial_value=36., volatility=0.1)
    vc.update('gbm2', initial_value=36., volatility=0.5)


if __name__ == '__main__':
    # valuation_mcs_euro_multi()
    # negative_correlation()
    pos_surface_corr()