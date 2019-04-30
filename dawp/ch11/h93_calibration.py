#
# Calibration of Bakshi, Cao and Chen (1997)
# Stoch Vol Jump Model to EURO STOXX Option Quotes
# Data Source: www.eurexchange.com
# via Numerical Integration
# 11_cal/BCC97_calibration_2.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import pdb
import sys
sys.path.append("/home/ubuntu/environment/python_for_finance/")
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.optimize import brute, fmin

from dawp.ch9.bcc_option_val import H93_call_value
from dawp.ch11.cir_calibration import CIR_calibration, r_list
from dawp.ch10.CIR_zcb_val_gen import B

DATA_PATH = '../data/'
mpl.rcParams['font.family'] = 'serif'
np.set_printoptions(suppress=True, formatter={'all': lambda x: '%5.3f' % x})


#
# Calibrate Short Rate Model
#
kappa_r, theta_r, sigma_r = CIR_calibration()

#
# Market Data from www.eurexchange.com
# as of 30. September 2014
#
h5 = pd.HDFStore(DATA_PATH + 'option_data.h5', 'r')
data = h5['data']  # European call & put option data (3 maturities)
h5.close()
S0 = 3225.93  # EURO STOXX 50 level 30.09.2014
r0 = r_list[0]  # initial short rate (Eonia 30.09.2014)

#
# Option Selection
#
tol = 0.02  # percent ITM/OTM options
options = data[(np.abs(data['Strike'] - S0) / S0) < tol]
options['Date'] = pd.DatetimeIndex(options['Date'])
options['Maturity'] = pd.DatetimeIndex(options['Maturity'])

# options = data[data['Strike'].isin([3100, 3150, 3225, 3300, 3350])]
#
# Adding Time-to-Maturity and Short Rates
#
for row, option in options.iterrows():
    T = (option['Maturity'] - option['Date']).days / 365.
    options.loc[row, 'T'] = T
    # pdb.set_trace()
    B0T = B([r0, kappa_r, theta_r, sigma_r, 0, T])
    options.loc[row, 'r'] = -math.log(B0T) / T

#
# Calibration Functions
#
i = 0
min_MSE = 500


def H93_error_function(p0):
    ''' Error function for parameter calibration in BCC97 model via
    Lewis (2001) Fourier approach.
    Parameters
    ==========
    kappa_v: float
        mean-reversion factor
    theta_v: float
        long-run mean of variance
    sigma_v: float
        volatility of variance
    rho: float
        correlation between variance and stock/index level
    v0: float
        initial, instantaneous variance
    Returns
    =======
    MSE: float
        mean squared error
    '''
    global i, min_MSE
    kappa_v, theta_v, sigma_v, rho, v0 = p0
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or \
            rho < -1.0 or rho > 1.0:
        return 500.0
    if 2 * kappa_v * theta_v < sigma_v ** 2:
        return 500.0
    se = []
    for row, option in options.iterrows():
        model_value = H93_call_value(S0, option['Strike'], option['T'],
                                     option['r'], kappa_v, theta_v, sigma_v,
                                     rho, v0)
        se.append((model_value - option['Call']) ** 2)
    MSE = sum(se) / len(se)
    min_MSE = min(min_MSE, MSE)
    if i % 25 == 0:
        print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (MSE, min_MSE))
    i += 1
    return MSE


def H93_calibration_full():
    ''' Calibrates H93 stochastic volatility model to market quotes. '''
    # first run with brute force
    # (scan sensible regions)
    p0 = brute(H93_error_function,
               ((2.5, 10.6, 5.0),  # kappa_v
                (0.01, 0.041, 0.01),  # theta_v
                (0.05, 0.251, 0.1),  # sigma_v
                (-0.75, 0.01, 0.25),  # rho
                (0.01, 0.031, 0.01)),  # v0
               finish=None)

    # second run with local, convex minimization
    # (dig deeper where promising)
    opt = fmin(H93_error_function, p0,
               xtol=0.000001, ftol=0.000001,
               maxiter=250, maxfun=500)
    # np.save('11_cal/opt_sv', np.array(opt))
    return opt


def c(p0):
    ''' Calculates all model values given parameter vector p0. '''
    kappa_v, theta_v, sigma_v, rho, v0 = p0
    values = []
    for row, option in options.iterrows():
        model_value = H93_call_value(S0, option['Strike'], option['T'],
                                     option['r'], kappa_v, theta_v, sigma_v,
                                     rho, v0)
        values.append(model_value)
    return np.array(values)
    
    
if __name__ == '__main__':
    pdb.set_trace()
    opt = H93_calibration_full()
    results = opt = H93_calibration_full()(opt)
    