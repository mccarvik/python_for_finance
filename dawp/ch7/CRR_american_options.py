#
# Valuation of American Options
# with the Cox-Ross-Rubinstein Model
# Primal Algorithm
# Case 1: American Put Option (APO)
# Case 2: Short Condor Spread (SCS)
# 07_amo/CRR_american_options.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math, pdb
import numpy as np

# General Parameters and Option Values


def set_parameters(otype, M):
    ''' Sets parameters depending on valuation case.
    Parameters
    ==========
    otype : int
        option type
        1 = American put option
        2 = Short Condor Spread
    '''
    if otype == 1:
        # Parameters -- American Put Option
        S0 = 36.  # initial stock level
        T = 1.0  # time-to-maturity
        r = 0.06  # short rate
        sigma = 0.2  # volatility

    elif otype == 2:
        # Parameters -- Short Condor Spread
        S0 = 100.  # initial stock level
        T = 1.0  # time-to-maturity
        r = 0.05  # short rate
        sigma = 0.5  # volatility

    else:
        raise ValueError('Option type not known.')

    # Numerical Parameters
    dt = T / M  # time interval
    df = math.exp(-r * dt)  # discount factor
    u = math.exp(sigma * math.sqrt(dt))  # up-movement
    d = 1 / u  # down-movement
    q = (math.exp(r * dt) - d) / (u - d)  # martingale probability

    return S0, T, r, sigma, M, dt, df, u, d, q


def inner_value(S, otype):
    ''' Inner value functions for American put option and short condor spread
    option with American exercise.
    Parameters
    ==========
    otype : int
        option type
        1 = American put option
        2 = Short Condor Spread
    '''
    if otype == 1:
        return np.maximum(40. - S, 0)
    elif otype == 2:
        return np.minimum(40., np.maximum(90. - S, 0) + np.maximum(S - 110., 0))
    else:
        raise ValueError('Option type not known.')


def CRR_option_valuation(otype, M=500):
    S0, T, r, sigma, M, dt, df, u, d, q = set_parameters(otype, M)
    # Array Generation for Stock Prices
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md

    # Valuation by Backwards Induction
    # Checking max of intrinsic value and exercising at each step
    h = inner_value(S, otype)  # innver value matrix
    V = inner_value(S, otype)  # value matrix
    C = np.zeros((M + 1, M + 1), dtype=np.float)  # continuation values
    ex = np.zeros((M + 1, M + 1), dtype=np.float)  # exercise matrix

    z = 0
    # Step backward from possible end values of S and calculate values at each node
    for i in range(M - 1, -1, -1):
        # C = continuation value, aka value of not xercising the option at this date
        # q is chance of up move --> (chance of up * up value in V + chance of down (1-q) * down value of V) * discount factor
        C[0:M - z, i] = (q * V[0:M - z, i + 1] + (1 - q) * V[1:M - z + 1, i + 1]) * df
        # V = present value of american derivative, max of C value and h where h is value of immediate exercise
        V[0:M - z, i] = np.where(h[0:M - z, i] > C[0:M - z, i], h[0:M - z, i], C[0:M - z, i])
        # ex simply states whether we exercise at each point or not
        ex[0:M - z, i] = np.where(h[0:M - z, i] > C[0:M - z, i], 1, 0)
        z += 1
    return V[0, 0]
    
if __name__ == '__main__':
    print(CRR_option_valuation(1))
    print(CRR_option_valuation(2))