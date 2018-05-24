import sys, pdb

import numpy as np
# import pandas as pd
import datetime as dt
from math import sqrt, pi, log, e

QUANDL_API_KEY = 'J4d6zKiPjebay-zW7T8X'

def derivative(f, x, h):
    return (f(x+h) - f(x-h)) / (2.0*h)  # might want to return a small non-zero if ==0


def newton_raphson(func, guess, rng=0.00001):
    try:
        lastX = guess
        nextX = lastX + 10* rng  # "different than lastX so loop starts OK
        while (abs(lastX - nextX) > rng):  # this is how you terminate the loop - note use of abs()
            newY = func(nextX)                     # just for debug... see what happens
            # print("f(", nextX, ") = ", newY)     # print out progress... again just debug
            lastX = nextX
            nextX = lastX - newY / derivative(func, lastX, rng)  # update estimate using N-R
        return nextX
    except Exception as e:
        pdb.set_trace()
        print()


def createCashFlows(start_dt, freq, mat_dt, cpn, par, par_cf=True):
    ''' Creats a list of tuple pairs where each pair is the date and amount of a cash flow
    Parameters
    ==========
    start_dt : date
        start_date of the calculation, usually today
    freq : float
        payment frequency
    mat_dt : date
        date of maturity of the bond
    cpn : float
        coupon rate, will be converted to dollar amount later
    par : float
        par amount of the bond at expiration
    
    Return
    ======
    cfs : list of tuples
        date, amount tuple pairs for each cash flow
    '''
    tenor = (mat_dt - start_dt).days / 365.25 # assumes 365.25 days in a year
    num_cfs = (1 / freq) * tenor
    days_from_issue = [int((365 * freq)*(i+1)) for i in range(round(num_cfs))]
    dates = [start_dt + dt.timedelta(i) for i in days_from_issue]
    cfs = [(dates[i], cpn * par * freq) for i in range(len(dates))]
    if par_cf:
        cfs.append((mat_dt, par))
    return cfs
    

def bootstrap(first_zero_rate, first_mat, bs_rate_mats):
    """ Will bootstrap the forward rates together to calculate the par rates. Using
    the previously calculated rate to discount for the next rate
    
    Parameters
    ==========
    first_zero_rate : float
        The first zero rate on the curve
    
    first_mat : date
        The maturity date of the first rate on the curve
    
    bs_rate_mats : list of tuples
        The rest of the rate, maturity pairs on the curve in the format (fwd_rate, maturity)

    Return
    ======
    list of tuples representing the par curve
    """
    pdb.set_trace()
    new_bs_rate_mats = []
    next_bs_zero_rate = ((bs_rate_mats[0][0] * bs_rate_mats[0][1]) + (first_zero_rate * first_mat)) / (bs_rate_mats[0][1] + first_mat)
    new_bs_rate_mats.append(tuple([bs_rate_mats[0][1] + first_mat, next_bs_zero_rate]))
    for i in range(len(bs_rate_mats)):
        if i == 0:
            continue
        
        next_bs_zero_rate = ((bs_rate_mats[i][0] * bs_rate_mats[i][1]) + \
            (new_bs_rate_mats[-1][1] * new_bs_rate_mats[-1][0])) / \
            (bs_rate_mats[i][1] + new_bs_rate_mats[-1][0])
        new_bs_rate_mats.append(tuple([bs_rate_mats[i][1] + new_bs_rate_mats[-1][0], next_bs_zero_rate]))
    return new_bs_rate_mats
    

def calcPV(cf, ir, period):
    ''' discounts the cash flow for the interest rate and period
    Parameters
    ==========
    cf : float
        amount of the cash flow
    ir : float
        current discount rate expressed in decimal terms
    period : float
        length of discount period
    
    Return
    ======
    cf : float
        discounted cash flow
    '''
    return (cf / (1+ir)**period)


def calcPVContinuous(cf, ir, period):
    ''' discounts the cash flow for the interest rate and period with continuous compounding
    Parameters
    ==========
    cf : float
        amount of the cash flow
    ir : float
        current discount rate
    period : float
        length of discount period
    
    Return
    ======
    cf : float
        discounted cash flow
    '''
    return cf * e**((-1) * ir * period)


def cumPresentValue(today, annual_disc_rate, cfs, freq=1, cont=True):
    ''' calculates the sum of the present value of all discounted cash flows
    Parameters
    ==========
    today : date
        trade date
    annual_disc_rate : float
        current annualized discount rate
    cfs : dataframe of tuples
        tuple pairs of amount and date for each cash flow
    freq : float
        frequency of payments
    cont : bool
        if the compounding is continuous or not
    
    Return
    ======
    cum_pv : float
        the cumulative present value of the cash flows
    '''
    # filters for only cash flows that haven't occurred yet
    cfs = [c for c in cfs if c[0] > today]
    
    cum_pv = 0
    # freq if rate passed in is not annual, ex: 0.5 = semiannual
    ir = annual_disc_rate * freq
    for i in cfs:
        period = ((i[0] - today).days / 365) / freq
        if cont:
            cum_pv += calcPVContinuous(i[1], ir, period)
        else:
            cum_pv += calcPV(i[1], ir, period)
    return cum_pv

    
if __name__ == '__main__':
    print(bootstrap(0.048, 400, [(0.053, 91), (0.055, 98)]))