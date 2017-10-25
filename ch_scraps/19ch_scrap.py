import sys, pdb
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'

import numpy as np
import pandas as pd
import datetime as dt
import calendar

from dx import *

import scipy.optimize as spo

PATH = '/home/ubuntu/workspace/python_for_finance/png/dx/'

i = 0

def volatility_options():
    
    # Hate these internal functions
    def third_friday(date):
        day = 21 - (calendar.weekday(date.year, date.month, 1) + 2) % 7
        return dt.datetime(date.year, date.month, day)


    def calculate_model_values(p0):
        ''' Returns all relevant option values.
        
        Parameters
        ===========
        p0 : tuple/list
            tuple of kappa, theta, volatility
        
        Returns
        =======
        model_values : dict
            dictionary with model values
        '''
        kappa, theta, volatility = p0
        vstoxx_model.update(kappa=kappa,
                            theta=theta,
                            volatility=volatility)
        model_values = {}
        for option in option_models:
           model_values[option] = \
             option_models[option].present_value(fixed_seed=True)
        return model_values
    
    i = 0
    def mean_squared_error(p0):
        ''' Returns the mean-squared error given
        the model and market values.
        
        Parameters
        ===========
        p0 : tuple/list
            tuple of kappa, theta, volatility
        
        Returns
        =======
        MSE : float
            mean-squared error
        '''
        global i
        model_values = np.array(list(calculate_model_values(p0).values()))
        market_values = option_selection['PRICE'].values
        option_diffs = model_values - market_values
        MSE = np.sum(option_diffs ** 2) / len(option_diffs)
          # vectorized MSE calculation
        if i % 20 == 0:
            if i == 0:
                print('%4s  %6s  %6s  %6s --> %6s' % 
                     ('i', 'kappa', 'theta', 'vola', 'MSE'))
            print('%4d  %6.3f  %6.3f  %6.3f --> %6.3f' % 
                    (i, p0[0], p0[1], p0[2], MSE))
        i += 1
        return MSE
    

    url = 'http://www.stoxx.com/download/historical_values/h_vstoxx.txt'
    vstoxx_index = pd.read_csv(url, index_col=0, header=2,
                           parse_dates=True, dayfirst=True,
                           sep=',')
    print(vstoxx_index.info())
    vstoxx_index = vstoxx_index[('2013/12/31' < vstoxx_index.index)
                            & (vstoxx_index.index < '2014/4/1')]
    print(np.round(vstoxx_index.tail(), 2))
    
    vstoxx_futures = pd.read_excel('./source/vstoxx_march_2014.xlsx',
                               'vstoxx_futures')
    print(vstoxx_futures.info())
    del vstoxx_futures['A_SETTLEMENT_PRICE_SCALED']
    del vstoxx_futures['A_CALL_PUT_FLAG']
    del vstoxx_futures['A_EXERCISE_PRICE']
    del vstoxx_futures['A_PRODUCT_ID']
    columns = ['DATE', 'EXP_YEAR', 'EXP_MONTH', 'PRICE']
    vstoxx_futures.columns = columns

    print(set(vstoxx_futures['EXP_MONTH']))
    third_fridays = {}
    for month in set(vstoxx_futures['EXP_MONTH']):
        third_fridays[month] = third_friday(dt.datetime(2014, month, 1))
    print(third_fridays)
    tf = lambda x: third_fridays[x]
    vstoxx_futures['MATURITY'] = vstoxx_futures['EXP_MONTH'].apply(tf)
    print(vstoxx_futures.tail())

    vstoxx_options = pd.read_excel('./source/vstoxx_march_2014.xlsx',
                                   'vstoxx_options')
    print(vstoxx_options.info())
    del vstoxx_options['A_SETTLEMENT_PRICE_SCALED']
    del vstoxx_options['A_PRODUCT_ID']
    columns = ['DATE', 'EXP_YEAR', 'EXP_MONTH', 'TYPE', 'STRIKE', 'PRICE']
    vstoxx_options.columns = columns
    vstoxx_options['MATURITY'] = vstoxx_options['EXP_MONTH'].apply(tf)
    print(vstoxx_options.head())
    vstoxx_options['STRIKE'] = vstoxx_options['STRIKE'] / 100.
    
    save = False
    if save is True:
        import warnings
        warnings.simplefilter('ignore')
        h5 = pd.HDFStore('./source/vstoxx_march_2014.h5',
                         complevel=9, complib='blosc')
        h5['vstoxx_index'] = vstoxx_index
        h5['vstoxx_futures'] = vstoxx_futures
        h5['vstoxx_options'] = vstoxx_options
        h5.close()

    pricing_date = dt.datetime(2014, 3, 31)
    # last trading day in March 2014
    maturity = third_fridays[10]
    # October maturity
    initial_value = vstoxx_index['V2TX'][pricing_date]
    # VSTOXX on pricing_date
    forward = vstoxx_futures[(vstoxx_futures.DATE == pricing_date)
                & (vstoxx_futures.MATURITY == maturity)]['PRICE'].values[0]
    tol = 0.20
    option_selection = vstoxx_options[(vstoxx_options.DATE == pricing_date)
                    & (vstoxx_options.MATURITY == maturity)
                    & (vstoxx_options.TYPE == 'C')
                    & (vstoxx_options.STRIKE > (1 - tol) * forward)
                    & (vstoxx_options.STRIKE < (1 + tol) * forward)]
    print(option_selection)
    me_vstoxx = market_environment('me_vstoxx', pricing_date)
    me_vstoxx.add_constant('initial_value', initial_value)
    me_vstoxx.add_constant('final_date', maturity)
    me_vstoxx.add_constant('currency', 'EUR')
    me_vstoxx.add_constant('frequency', 'B')
    me_vstoxx.add_constant('paths', 1000)
    csr = constant_short_rate('csr', 0.01)
    # somewhat arbitrarily chosen here
    me_vstoxx.add_curve('discount_curve', csr)
    # parameters to be calibrated later
    me_vstoxx.add_constant('kappa', 1.0)
    me_vstoxx.add_constant('theta', 1.2 * initial_value)
    vol_est =  vstoxx_index['V2TX'].std() \
                * np.sqrt(len(vstoxx_index['V2TX']) / 252.)
    me_vstoxx.add_constant('volatility', vol_est)
    print(vol_est)
    
    vstoxx_model = square_root_diffusion('vstoxx_model', me_vstoxx)
    me_vstoxx.add_constant('strike', forward)
    me_vstoxx.add_constant('maturity', maturity)
    payoff_func = 'np.maximum(maturity_value - strike, 0)'
    vstoxx_eur_call = valuation_mcs_european_single('vstoxx_eur_call',
                            vstoxx_model, me_vstoxx, payoff_func)
    print(vstoxx_eur_call.present_value())
    option_models = {}
    for option in option_selection.index:
        strike = option_selection['STRIKE'].ix[option]
        me_vstoxx.add_constant('strike', strike)
        option_models[option] = valuation_mcs_european_single(
                                'eur_call_%d' % strike,
                                vstoxx_model,
                                me_vstoxx,
                                payoff_func)


    print(calculate_model_values((0.5, 27.5, vol_est)))
    i = 0

    print(mean_squared_error((0.5, 27.5, vol_est)))

    i = 0
    opt_global = spo.brute(mean_squared_error,
                    ((0.5, 3.01, 0.5),  # range for kappa
                     (15., 30.1, 5.),  # range for theta
                     (0.5, 5.51, 1)),  # range for volatility
                     finish=None)
    print(opt_global)
    i = 0
    print(mean_squared_error(opt_global))
    
    i = 0
    opt_local = spo.fmin(mean_squared_error, opt_global,
                         xtol=0.00001, ftol=0.00001,
                         maxiter=100, maxfun=350)
    print(opt_local)

    i = 0
    print(mean_squared_error(opt_local))

    print(calculate_model_values(opt_local))

    pd.options.mode.chained_assignment = None
    option_selection['MODEL'] = \
            np.array(list(calculate_model_values(opt_local).values()))
    option_selection['ERRORS'] = \
            option_selection['MODEL'] - option_selection['PRICE']
    print(option_selection[['MODEL', 'PRICE', 'ERRORS']])
    print(round(option_selection['ERRORS'].mean(), 3))

    fix, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8, 8))
    strikes = option_selection['STRIKE'].values
    ax1.plot(strikes, option_selection['PRICE'], label='market quotes')
    ax1.plot(strikes, option_selection['MODEL'], 'ro', label='model values')
    ax1.set_ylabel('option values')
    ax1.grid(True)
    ax1.legend(loc=0)
    wi = 0.25
    ax2.bar(strikes - wi / 2., option_selection['ERRORS'],
            label='market quotes', width=wi)
    ax2.grid(True)
    ax2.set_ylabel('differences')
    ax2.set_xlabel('strikes')
    plt.savefig(PATH + 'vstoxx_options.png', dpi=300)
    plt.close()

    me_vstoxx = market_environment('me_vstoxx', pricing_date)
    me_vstoxx.add_constant('initial_value', initial_value)
    me_vstoxx.add_constant('final_date', pricing_date)
    me_vstoxx.add_constant('currency', 'NONE')

    # adding optimal parameters to environment
    me_vstoxx.add_constant('kappa', opt_local[0])
    me_vstoxx.add_constant('theta', opt_local[1])
    me_vstoxx.add_constant('volatility', opt_local[2])
    me_vstoxx.add_constant('model', 'srd')
    payoff_func = 'np.maximum(strike - instrument_values, 0)'
    shared = market_environment('share', pricing_date)
    shared.add_constant('maturity', maturity)
    shared.add_constant('currency', 'EUR')
    option_positions = {}
    # dictionary for option positions
    option_environments = {}
    # dictionary for option environments
    for option in option_selection.index:
        option_environments[option] = \
            market_environment('am_put_%d' % option, pricing_date)
        # define new option environment, one for each option
        strike = option_selection['STRIKE'].ix[option]
        # pick the relevant strike
        option_environments[option].add_constant('strike', strike)
        # add it to the environment
        option_environments[option].add_environment(shared)
        # add the shared data
        option_positions['am_put_%d' % strike] = \
                        derivatives_position(
                            'am_put_%d' % strike,
                            quantity=100.,
                            underlyings=['vstoxx_model'],
                            mar_env=option_environments[option],
                            otype='American single',
                            payoff_func=payoff_func)

    val_env = market_environment('val_env', pricing_date)
    val_env.add_constant('starting_date', pricing_date)
    val_env.add_constant('final_date', pricing_date)
    # temporary value, is updated during valuation
    val_env.add_curve('discount_curve', csr)
    val_env.add_constant('frequency', 'B')
    val_env.add_constant('paths', 2500)
    underlyings = {'vstoxx_model' : me_vstoxx}

    pdb.set_trace()
    portfolio = derivatives_portfolio('portfolio', option_positions,
                                      val_env, underlyings)
    results = portfolio.get_statistics(fixed_seed=True)
    print(results.sort_values(by='name'))

    print(results[['pos_value','pos_delta','pos_vega']].sum())



    
if __name__ == '__main__': 
    volatility_options()