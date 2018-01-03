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

PATH = '/home/ubuntu/workspace/python_for_finance/png/scrap/models/'

colormap='RdYlBu_r'
lw=1.25
figsize=(10, 6)
legend=False
no_paths=10

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


def geometric_brownian_motion_run():
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
    plt.close()
    
    gbm.update(volatility=0.5)
    paths_2 = gbm.get_instrument_values()
    plt.figure(figsize=(8, 4))
    p1 = plt.plot(gbm.time_grid, paths_1[:, :10], 'b')
    p2 = plt.plot(gbm.time_grid, paths_2[:, :10], 'r-.')
    plt.grid(True)
    l1 = plt.legend([p1[0], p2[0]],
                ['low volatility', 'high volatility'], loc=2)
    plt.gca().add_artist(l1)
    plt.xticks(rotation=30)
    plt.savefig(PATH + 'gbm_vols.png', dpi=300)
    plt.close()


def jump_diffusion_run():
    yields = [0.0025, 0.01, 0.015, 0.025]
    dates = [dt.datetime(2015, 1, 1), dt.datetime(2016, 1, 1), dt.datetime(2020, 1, 1), dt.datetime(2025, 1, 1)]
    dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))
    
    me_jd = market_environment('me_jd', dt.datetime(2015, 1, 1))
    # specific to simulation class
    me_jd.add_constant('lambda', 0.3)       # probability for a jump p.a.
    me_jd.add_constant('mu', -0.75)         # expected relative jump size
    me_jd.add_constant('delta', 0.1)        # standard deviation of relative jump
    # from GBM environment
    me_jd.add_constant('initial_value', 100.)
    me_jd.add_constant('volatility', 0.2)
    me_jd.add_constant('final_date', dt.datetime(2015, 12, 31))
    me_jd.add_constant('currency', 'EUR')
    # monthly frequency (respcective month end)
    me_jd.add_constant('frequency', 'M')
    me_jd.add_constant('paths', 1)
    me_jd.add_curve('discount_curve', dsr)
    
    jd = jump_diffusion('jd', me_jd)
    jd.generate_time_grid()
    paths_3 = jd.get_instrument_values(fixed_seed=False)
    jd.update(lamb=0.9)
    paths_4 = jd.get_instrument_values(fixed_seed=False)
    plt.figure(figsize=(8, 4))
    p1 = plt.plot(jd.time_grid, paths_3[:, :10], 'b')
    p2 = plt.plot(jd.time_grid, paths_4[:, :10], 'r-.')
    plt.grid(True)
    l1 = plt.legend([p1[0], p2[0]],
                ['low intensity', 'high intensity'], loc=3)
    plt.gca().add_artist(l1)
    plt.xticks(rotation=30)
    plt.savefig(PATH + 'jd.png', dpi=300)
    plt.close()


def stoch_volatility():
    yields = [0.0025, 0.01, 0.015, 0.025]
    dates = [dt.datetime(2015, 1, 1), dt.datetime(2016, 1, 1), dt.datetime(2020, 1, 1), dt.datetime(2025, 1, 1)]
    dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))
    
    no_paths = 5
    
    # Market Environment setup
    me = market_environment('me', dt.datetime(2015, 1, 1))
    me.add_constant('initial_value', 36.)
    me.add_constant('volatility', 0.2)
    me.add_constant('final_date', dt.datetime(2016, 12, 31))
    me.add_constant('currency', 'EUR')
    me.add_constant('frequency', 'M')       # monthly frequency; paramter accorind to pandas convention
    me.add_constant('paths', no_paths)
    me.add_curve('discount_curve', dsr)
    me.add_constant('rho', -.5)             # correlation between risk factor process (eg index) and variance process 
    me.add_constant('kappa', 2.5)           # mean reversion factor
    me.add_constant('theta', 0.1)           # long-term variance level
    me.add_constant('vol_vol', 0.1)         # volatility factor for variance process
    
    # Monte carlo with stochastic volatility
    # sv = stochastic_volatility('sv', me)
    # paths = sv.get_instrument_values(fixed_seed=False)
    # pdf = pd.DataFrame(paths, index=sv.time_grid)
    # pdf[pdf.columns[:no_paths]].plot(colormap=colormap, lw=lw, figsize=figsize, legend=legend)
    # plt.savefig(PATH + 'sv_paths.png', dpi=300)
    # plt.close()
    
    # vols = sv.get_volatility_values()
    # pdf = pd.DataFrame(vols, index=sv.time_grid)
    # pdf[pdf.columns[:no_paths]].plot(colormap=colormap, lw=lw, figsize=figsize, legend=legend)
    # plt.savefig(PATH + 'sv_vol.png', dpi=300)
    # plt.close()
    
    me.add_constant('lambda', 0.3)       # probability for a jump
    me.add_constant('mu', -0.75)         # expected relative jump size
    me.add_constant('delta', 0.1)        # standard deviation of relative jump
    
    # Already have all the market parameters we need for stochastic vol with jump diffusion 
    svjd = stoch_vol_jump_diffusion('svjd', me)
    paths = svjd.get_instrument_values(fixed_seed=False)
    pdf = pd.DataFrame(paths, index=svjd.time_grid)
    pdf[pdf.columns[:no_paths]].plot(colormap=colormap, lw=lw, figsize=figsize, legend=legend)
    plt.savefig(PATH + 'svjd_paths.png', dpi=300)
    plt.close()
    
    vols = svjd.get_volatility_values()
    pdf = pd.DataFrame(vols, index=svjd.time_grid)
    pdf[pdf.columns[:no_paths]].plot(colormap=colormap, lw=lw, figsize=figsize, legend=legend)
    plt.savefig(PATH + 'svjd_vol.png', dpi=300)
    plt.close()
    

def sabr_stoch_vol():
    yields = [0.0025, 0.01, 0.015, 0.025]
    dates = [dt.datetime(2015, 1, 1), dt.datetime(2016, 1, 1), dt.datetime(2020, 1, 1), dt.datetime(2025, 1, 1)]
    dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))
    
    no_paths = 5

    # Market Environment setup
    me = market_environment('me', dt.datetime(2015, 1, 1))
    me.add_constant('initial_value', 50)
    me.add_constant('alpha', 0.04)          # initial variance, vairance = vol^2
    me.add_constant('beta', 0.5)            # exponent, 0 <= beta <= 1
    me.add_constant('rho', 0.1)             # correlation factor
    me.add_constant('volatility', 0.2)
    me.add_constant('vol_vol', 0.5)         # volatility of volatility/variance
    
    me.add_constant('final_date', dt.datetime(2020, 12, 31))
    me.add_constant('currency', 'EUR')
    me.add_constant('frequency', 'M')       # M = monthly frequency; paramter accorind to pandas convention
    me.add_constant('paths', no_paths)
    me.add_curve('discount_curve', dsr)
    
    sabr = sabr_stochastic_volatility('sabr', me)
    paths = sabr.get_instrument_values(fixed_seed=False)
    pdf = pd.DataFrame(paths, index=sabr.time_grid)
    pdf[pdf.columns[:no_paths]].plot(colormap=colormap, lw=lw, figsize=figsize, legend=legend)
    plt.savefig(PATH + 'sabr_paths.png', dpi=300)
    plt.close()
    
    vols = sabr.get_volatility_values()
    pdf = pd.DataFrame(vols, index=sabr.time_grid)
    pdf[pdf.columns[:no_paths]].plot(colormap=colormap, lw=lw, figsize=figsize, legend=legend)
    plt.savefig(PATH + 'sabr_vol.png', dpi=300)
    plt.close()
    

def mean_revert_diff():
    yields = [0.0025, 0.01, 0.015, 0.025]
    dates = [dt.datetime(2015, 1, 1), dt.datetime(2016, 1, 1), dt.datetime(2020, 1, 1), dt.datetime(2025, 1, 1)]
    dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))
    
    # no_paths = 5
    no_paths = 1

    # Market Environment setup
    me = market_environment('me', dt.datetime(2015, 1, 1))
    me.add_constant('initial_value', 0.05)
    me.add_constant('kappa', 2.5)          # initial variance, vairance = vol^2
    me.add_constant('theta', 0.01)            # exponent, 0 <= beta <= 1
    me.add_constant('volatility', 0.05)
    
    me.add_constant('final_date', dt.datetime(2020, 12, 31))
    me.add_constant('currency', 'EUR')
    me.add_constant('frequency', 'A')       # M = monthly frequency; paramter accorind to pandas convention
    me.add_constant('paths', no_paths)
    me.add_curve('discount_curve', dsr)
    
    mrd = mean_reverting_diffusion('mrd', me)
    paths = mrd.get_instrument_values(fixed_seed=False)
    pdf = pd.DataFrame(paths, index=mrd.time_grid)
    pdf[pdf.columns[:no_paths]].plot(colormap=colormap, lw=lw, figsize=figsize, legend=legend)
    plt.savefig(PATH + 'mean_revert_diff.png', dpi=300)
    plt.close()


def sqrt_diffusion():
    yields = [0.0025, 0.01, 0.015, 0.025]
    dates = [dt.datetime(2015, 1, 1), dt.datetime(2016, 1, 1), dt.datetime(2020, 1, 1), dt.datetime(2025, 1, 1)]
    dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))
    
    # no_paths = 5
    no_paths = 1
    
    # Market Environment setup
    me = market_environment('me', dt.datetime(2015, 1, 1))
    me.add_constant('initial_value', 0.05)
    me.add_constant('volatility', 0.05)
    me.add_constant('final_date', dt.datetime(2020, 12, 31))
    me.add_constant('currency', 'EUR')
    me.add_constant('frequency', 'A')       # M = monthly frequency; paramter accorind to pandas convention
    me.add_constant('paths', no_paths)
    me.add_curve('discount_curve', dsr)
    
    # Model specific variables
    me.add_constant('kappa', 0.5)          # initial variance, vairance = vol^2
    me.add_constant('theta', 0.1)            # exponent, 0 <= beta <= 1
    
    srd = square_root_diffusion('srd', me)
    paths = srd.get_instrument_values(fixed_seed=False)
    pdf = pd.DataFrame(paths, index=srd.time_grid)
    pdf[pdf.columns[:no_paths]].plot(colormap=colormap, lw=lw, figsize=figsize, legend=legend)
    plt.savefig(PATH + 'sqrt_diffusion.png', dpi=300)
    plt.close()


def sqrt_jump_diffusion():
    yields = [0.0025, 0.01, 0.015, 0.025]
    dates = [dt.datetime(2015, 1, 1), dt.datetime(2016, 1, 1), dt.datetime(2020, 1, 1), dt.datetime(2025, 1, 1)]
    dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))
    
    # no_paths = 5
    no_paths = 1
    
    # Market Environment setup
    me = market_environment('me', dt.datetime(2015, 1, 1))
    me.add_constant('initial_value', 25)
    me.add_constant('volatility', 0.05)
    me.add_constant('final_date', dt.datetime(2020, 12, 31))
    me.add_constant('currency', 'EUR')
    me.add_constant('frequency', 'A')       # M = monthly frequency; paramter accorind to pandas convention
    me.add_constant('paths', no_paths)
    me.add_curve('discount_curve', dsr)
    
    # Model specific variables
    me.add_constant('kappa', 0.5)          # initial variance, vairance = vol^2
    me.add_constant('theta', 20)            # exponent, 0 <= beta <= 1
    me.add_constant('lambda', 0.3)       # probability for a jump p.a.
    me.add_constant('mu', -0.75)         # expected relative jump size
    me.add_constant('delta', 0.1)        # standard deviation of relative jump
    
    srd = square_root_jump_diffusion('srd', me)
    paths = srd.get_instrument_values(fixed_seed=False)
    pdf = pd.DataFrame(paths, index=srd.time_grid)
    pdf[pdf.columns[:no_paths]].plot(colormap=colormap, lw=lw, figsize=figsize, legend=legend)
    plt.savefig(PATH + 'sqrt_jump_diffusion.png', dpi=300)
    plt.close()


def sqrt_jump_diffusion_plus():
    yields = [0.0025, 0.01, 0.015, 0.025]
    dates = [dt.datetime(2015, 1, 1), dt.datetime(2016, 1, 1), dt.datetime(2020, 1, 1), dt.datetime(2025, 1, 1)]
    dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))
    
    no_paths = 10
    # no_paths = 1
    
    # Market Environment setup
    me = market_environment('me', dt.datetime(2015, 1, 1))
    me.add_constant('initial_value', 25)
    me.add_constant('volatility', 0.05)
    me.add_constant('final_date', dt.datetime(2020, 12, 31))
    me.add_constant('currency', 'EUR')
    me.add_constant('frequency', 'A')       # M = monthly frequency; paramter accorind to pandas convention
    me.add_constant('paths', no_paths)
    me.add_curve('discount_curve', dsr)
    
    # Model specific variables
    me.add_constant('kappa', 0.5)          # initial variance, vairance = vol^2
    me.add_constant('theta', 20)            # exponent, 0 <= beta <= 1
    me.add_constant('lambda', 0.3)       # probability for a jump p.a.
    me.add_constant('mu', -0.75)         # expected relative jump size
    me.add_constant('delta', 0.1)        # standard deviation of relative jump
    
    # Add volatility term structure
    term_structure = np.array([(dt.datetime(2015, 1, 1), 25.),
                  (dt.datetime(2016, 3, 31), 24.),
                  (dt.datetime(2017, 6, 30), 27.),
                  (dt.datetime(2018, 9, 30), 28.),
                  (dt.datetime(2020, 12, 31), 30.)])
    me.add_curve('term_structure', term_structure)
    
    # TODO: need to step thru this
    srd = square_root_jump_diffusion_plus('srd', me)
    
    # calibrates the square-root diffusion to the given term structure
    srd.generate_shift_base((2.0, 20., 0.1))
    # i.e. the difference between the model and market implied foward rates
    print("Shift Base:")
    print(srd.shift_base)
    
    # calculates deterministic shift values for the relevant time grid by interpolation of the shift_base values
    print("Shift Values:")
    srd.update_shift_values()
    print(srd.shift_values)
    
    # model forward rates resulting from parameters and adjustments
    print("Updated Fwd Rates:")
    srd.update_forward_rates()
    print(srd.forward_rates)
    
    paths = srd.get_instrument_values(fixed_seed=False)
    pdf = pd.DataFrame(paths, index=srd.time_grid)
    pdf[pdf.columns[:no_paths]].plot(colormap=colormap, lw=lw, figsize=figsize, legend=legend)
    plt.savefig(PATH + 'sqrt_jump_diffusion_plus.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    # risk_neutral_discounting()
    # creating_market_environment()
    # standard_normal_random_numbers()
    # geometric_brownian_motion_run()
    # jump_diffusion_run()
    # stoch_volatility()
    # sabr_stoch_vol()
    # mean_revert_diff()
    # sqrt_diffusion()
    # sqrt_jump_diffusion()
    sqrt_jump_diffusion_plus()