import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
# sys.path.append("/usr/local/lib/python3.4/dist-packages")
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

from dx import *
from utils.utils import *

import numpy as np
# import pandas as pd
import datetime as dt

# yields = [0.0025, 0.01, 0.015, 0.025]
# dates = [dt.datetime(2015, 1, 1), dt.datetime(2015, 7, 1), dt.datetime(2015, 12, 31), dt.datetime(2018, 12, 31)]
yields = [0.01, 0.03, 0.04, 0.046, 0.05, 0.053]
dates = [dt.datetime(2015,1,1), dt.datetime(2016,1,1), dt.datetime(2017,1,1), dt.datetime(2018,1,1), dt.datetime(2019,1,1), dt.datetime(2020,1,1)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))


def compounding_conversion(r, yrs, prds=1, cont=False):
    if not cont:
        return (1 + r/prds)**(prds*yrs)
    else:
        return np.exp(r*yrs)


def convert_to_continuous(r, prds):
    return prds * np.log(1 + r/prds)


def convert_from_continuous(r, prds):
    return prds * (np.exp(r/prds) - 1)
    
    
def calc_continuous_zero_rate(price, par, settle_dt, mat_dt):
    return (-1) * (np.log(price/par)/ get_year_deltas([settle_dt, mat_dt])[-1])


# Same as Bond Yield - Calculates the bond yield for a bond to equal its market price
def calc_yld_to_date(price, par, settle_dt, mat_dt, cpn, freq=0.5, guess=None, cont_disc=False):
    ''' Takes a price and a cpn rate and then uses a newton-raphson approximation to
        zero in on the interest rate (i.e. YTM) that resolves the equation of all the discounted
        cash flows within and reasonable range
    Parameters
    ==========
    start_date : date
        start_date of the calculation, usually today
    freq : float
        payment frequency
    mat_date : date
        date of maturity of the bond
    cpn : float
        coupon rate
    par : float
        par amount of the bond at expiration
    price : float
        given price of the bond
    guess : float
        used for newton raphson approximation so the equation conforms quicker, defaults to the cpn rate
    cont_dics : bool
        used to decide if we want to use continuous discounting or not
    
    Return
    ======
    ytm : float
        returns the calculated approximate YTM
    '''
    tenor = (mat_dt - settle_dt).days / 365.25 # assumes 365.25 days in a year
    freq = float(freq)
    # guess ytm = coupon rate, will get us in the ball park
    guess = cpn
    # convert cpn from annual rate to actual coupon value recieved
    coupon = cpn * freq * par
    cfs = createCashFlows(settle_dt, freq, mat_dt, cpn, par)
    # filters for only cash flows that haven't occurred yet
    cfs = [c for c in cfs if c[0] > settle_dt]
    cpn_dts = [((i[0] - settle_dt).days / 365, i[1]) for i in cfs]
    
    if not cont_disc:
        ytm_func = lambda y: sum([c/(1+y*freq)**(t/freq) for t,c in cpn_dts]) - price
    else:
        ytm_func = lambda y: sum([c*np.exp(-y*t) for t,c in cpn_dts]) - price
        
    return newton_raphson(ytm_func, guess)


# Calculates the coupon a bond would need to trade at par given a yield curve
def calc_par_yield(par, settle_dt, mat_dt, freq):
    tenor = (mat_dt - settle_dt).days / 365.25 # assumes 365.25 days in a year
    freq = float(freq)
    # guess ytm = coupon rate, will get us in the ball park
    guess = dsr.yield_list[0][1]
    # convert cpn from annual rate to actual coupon value recieved
    cfs = createCashFlows(settle_dt, freq, mat_dt, 0, par)
    # filters for only cash flows that haven't occurred yet
    cfs = [c for c in cfs if c[0] > settle_dt]
    cpn_dts = [((i[0] - settle_dt).days / 365, i[1]) for i in cfs]
    
    # swap in disc_rate for coupon
    disc_dts = [(dtt, dsr.get_interpolated_yields([settle_dt, settle_dt + dt.timedelta(days=365*dtt)])[-1][1]) for dtt, cpn in cpn_dts]
    
    # NOTE: the "-2" on the index removes the discount for the last coupon and par payment to be dealt with in the
    # second half of the equation. May need to adjust this depending on the bond
    par_yld_func = lambda y: sum([(y*freq)*np.exp(-d*t) for t,d in disc_dts[:-2]]) + \
                            (par + (y*freq))*np.exp(-1*disc_dts[-1][0]*disc_dts[-1][1]) - par
    
    return newton_raphson(par_yld_func, guess)
    

# Calculates the zero rate using bootstrapping
def calc_zero_rate(price, par, cpn, prds, settle_dt, mat_dt, rate_mat_pairs):
    cpn = cpn / prds * par
    print([get_year_deltas([settle_dt, dt])[-1] for r, dt in rate_mat_pairs])
    prev_sum = sum([cpn * np.exp(-r * get_year_deltas([settle_dt, dt])[-1]) for r, dt in rate_mat_pairs])
    price = price - prev_sum
    par = par + cpn
    return calc_continuous_zero_rate(price, par, settle_dt, mat_dt)
    
    
# Get fwd rate from two dates and a spot curve
def calc_fwd_rate(settle_dt, fwd_dt, mat_dt):
    ylds = dsr.get_interpolated_yields([settle_dt, fwd_dt, mat_dt])
    r2 = ylds[-1][1]
    r1 = ylds[-2][1]
    dts = get_year_deltas([settle_dt, fwd_dt, mat_dt])
    dt2 = dts[-1]
    dt1 = dts[-2]
    return (r2*dt2 - r1*dt1) / (dt2 - dt1)


# def fra_valuation_cont_dsic(orig_rate, settle_dt, fwd_dt, mat_dt):
#     num = dsr.get_interpolated_yields([settle_dt, fwd_dt, mat_dt])
#     yld = dsr.get_interpolated_yields([settle_dt, fwd_dt, mat_dt])
    
    
    

if __name__ == '__main__':
    # print(compounding_conversion(.10, 2, 2))
    # print(compounding_conversion(.10, 2, cont=True))
    # print(convert_to_continuous(.10, 2))
    # print(convert_from_continuous(.08, 4))
    # print(calc_yld_to_date(98.39, 100, dt.datetime(2015,1,1), dt.datetime(2016,12,31), 0.06, cont_disc=True))
    # print(calc_par_yield(100, dt.datetime(2016,1,1), dt.datetime(2017,12,31), 0.5))
    
    # print(calc_continuous_zero_rate(97.5, 100, dt.datetime(2017,1,1), dt.datetime(2017,4,1)))
    # rate_mats = [(0.10469, dt.datetime(2015,7,1)), (0.10536, dt.datetime(2016,1,1)), (0.10681, dt.datetime(2016,7,1))]
    # print(calc_zero_rate(101.6, 100, 0.12, 2, dt.datetime(2015,1,1), dt.datetime(2017,1,1), rate_mats))
    
    print(calc_fwd_rate(dt.datetime(2015,1,1), dt.datetime(2018,1,1), dt.datetime(2019,1,1)))
    print(fra_valuation_cont_dsic(dt.datetime(2015,1,1), dt.datetime(2018,1,1), dt.datetime(2019,1,1)))