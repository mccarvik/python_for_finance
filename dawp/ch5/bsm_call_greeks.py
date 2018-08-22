#
# Black-Scholes-Merton (1973) European Call Option Greeks
# 05_com/BSM_call_greeks.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from bsm_option_valuation import d1f, N, dN
mpl.rcParams['font.family'] = 'serif'

PNG_PATH = '../png/5ch/'

#
# Functions for Greeks
#


def BSM_delta(St, K, t, T, r, sigma):
    # change in option price to change in underlying
    ''' Black-Scholes-Merton DELTA of European call option.
    Parameters
    ==========
    St : float
        stock/index level at time t
    K : float
        strike price
    t : float
        valuation date
    T : float
        date of maturity/time-to-maturity if t = 0; T > t
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    Returns
    =======
    delta : float
        European call option DELTA
    '''
    d1 = d1f(St, K, t, T, r, sigma)
    delta = N(d1)
    return delta


def BSM_gamma(St, K, t, T, r, sigma):
    # change in delta to change in underlying
    ''' Black-Scholes-Merton GAMMA of European call option.
    Parameters
    ==========
    St : float
        stock/index level at time t
    K : float
        strike price
    t : float
        valuation date
    T : float
        date of maturity/time-to-maturity if t = 0; T > t
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    Returns
    =======
    gamma : float
        European call option GAMM
    '''
    d1 = d1f(St, K, t, T, r, sigma)
    gamma = dN(d1) / (St * sigma * math.sqrt(T - t))
    return gamma


def BSM_theta(St, K, t, T, r, sigma):
    # change in option price to change in time to maturity
    ''' Black-Scholes-Merton THETA of European call option.
    Parameters
    ==========
    St : float
        stock/index level at time t
    K : float
        strike price
    t : float
        valuation date
    T : float
        date of maturity/time-to-maturity if t = 0; T > t
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    Returns
    =======
    theta : float
        European call option THETA
    '''
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    theta = -(St * dN(d1) * sigma / (2 * math.sqrt(T - t)) +
              r * K * math.exp(-r * (T - t)) * N(d2))
    return theta


def BSM_rho(St, K, t, T, r, sigma):
    # change in option price to change in interest rate
    ''' Black-Scholes-Merton RHO of European call option.
    Parameters
    ==========
    St : float
        stock/index level at time t
    K : float
        strike price
    t : float
        valuation date
    T : float
        date of maturity/time-to-maturity if t = 0; T > t
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    Returns
    =======
    rho : float
        European call option RHO
    '''
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    rho = K * (T - t) * math.exp(-r * (T - t)) * N(d2)
    return rho


def BSM_vega(St, K, t, T, r, sigma):
    # change in option price to change in volatility
    ''' Black-Scholes-Merton VEGA of European call option.
    Parameters
    ==========
    St : float
        stock/index level at time t
    K : float
        strike price
    t : float
        valuation date
    T : float
        date of maturity/time-to-maturity if t = 0; T > t
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    Returns
    =======
    vega : float
        European call option VEGA
    '''
    d1 = d1f(St, K, t, T, r, sigma)
    vega = St * dN(d1) * math.sqrt(T - t)
    return vega

#
# Plotting the Greeks
#


def plot_greeks(function, greek):
    # Model Parameters
    St = 100.0  # index level
    r = 0.05  # risk-less short rate
    sigma = 0.2  # volatility
    t = 0.0  # valuation date

    # Greek Calculations
    tlist = np.linspace(0.01, 1, 25)
    klist = np.linspace(80, 120, 25)
    V = np.zeros((len(tlist), len(klist)), dtype=np.float)
    for j in range(len(klist)):
        for i in range(len(tlist)):
            V[i, j] = function(St, klist[j], t, tlist[i], r, sigma)

    # 3D Plotting
    x, y = np.meshgrid(klist, tlist)
    fig = plt.figure(figsize=(9, 5))
    plot = p3.Axes3D(fig)
    plot.plot_wireframe(x, y, V)
    plot.set_xlabel('strike $K$')
    plot.set_ylabel('maturity $T$')
    plot.set_zlabel('%s(K, T)' % greek)
    plt.savefig(PNG_PATH + function.__name__, dpi=300)
    plt.close()
    

if __name__ == '__main__':
    plot_greeks(BSM_delta, 'delta')
    plot_greeks(BSM_gamma, 'gamma')
    plot_greeks(BSM_theta, 'theta')
    plot_greeks(BSM_rho, 'rho')
    plot_greeks(BSM_vega, 'vega')
    
    