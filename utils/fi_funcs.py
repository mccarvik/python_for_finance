import sys
sys.path.append("/home/ubuntu/workspace/finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import datetime, sys, pdb
from scipy import optimize
from math import sqrt, pi, log, e
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
# from app.curves.curves import Curve


FREQ_MAP = {
    'Semi-Annual' : 0.5,
    'Quarterly' : 0.25
}

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


def createCashFlows(start_date, freq, mat_date, cpn, par, par_cf=True):
    ''' Creats a list of tuple pairs where each pair is the date and amount of a cash flow
    Parameters
    ==========
    start_date : date
        start_date of the calculation, usually today
    freq : float
        payment frequency
    mat_date : date
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
    tenor = (mat_date - start_date).days / 365.25 # assumes 365.25 days in a year
    num_cfs = (1 / freq) * tenor
    days_from_issue = [int((365 * freq)*(i+1)) for i in range(round(num_cfs))]
    dates = [start_date + datetime.timedelta(i) for i in days_from_issue]
    cfs = [(dates[i], cpn * par * freq) for i in range(len(dates))]
    if par_cf:
        cfs.append((mat_date, par))
    return cfs


def calcYieldToDate(price, par, mat_date, cpn, freq=0.5, start_date=datetime.datetime.today(), guess=None):
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
    
    Return
    ======
    ytm : float
        returns the calculated approximate YTM
    '''
    tenor = (mat_date - start_date).days / 365.25 # assumes 365.25 days in a year
    freq = float(freq)
    # guess ytm = coupon rate, will get us in the ball park
    guess = cpn
    # convert cpn from annual rate to actual coupon value recieved
    coupon = cpn * freq * par
    cfs = createCashFlows(start_date, freq, mat_date, cpn, par)
    # filters for only cash flows that haven't occurred yet
    cfs = [c for c in cfs if c[0] > start_date]
    cpn_dts = [((i[0] - start_date).days / 365, i[1]) for i in cfs]
    
    ytm_func = lambda y: \
        sum([c/(1+y*freq)**(t/freq) for t,c in cpn_dts]) - price
        
    # return optimize.newton(ytm_func, guess)
    return newton_raphson(ytm_func, guess)


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
    except:
        import pdb; pdb.set_trace()
        print()
    

def calcSurvivalRate(time, rate):
    # time measured in years
    return (e**(-1*time*rate))


def VaR(symbol='AAPL', notl=None, conf=0.95, dist=None, _d1=None, _d2=None, volwindow=50, varwindow=250):
    # Retrieve the data from Internet
    # Choose a time period
    d1 = _d1 if _d1 else datetime.datetime(2001, 1, 1)
    d2 = _d2 if _d2 else datetime.datetime(2012, 1, 1)
    #get the tickers
    price = DataReader(symbol, "yahoo",d1,d2)['Adj Close']
    price = price.asfreq('B').fillna(method='pad')
    ret = price.pct_change()
    
    #choose the quantile
    quantile=1-conf
    
    import pdb; pdb.set_trace()
    #simple VaR using all the data
    # VaR on average accross all the data
    unnormedquantile=pd.expanding_quantile(ret,quantile)
    
    # similar one using a rolling window 
    # VaR only calculated over the varwindow, rolling
    unnormedquantileR=pd.rolling_quantile(ret,varwindow,quantile)
    
    #we can also normalize the returns by the vol
    vol=pd.rolling_std(ret,volwindow)*np.sqrt(256)
    unitvol=ret/vol
    
    #and get the expanding or rolling quantiles
    # Same calcs as above except normalized so show VaR in
    # standard deviations instead of expected returns
    Var=pd.expanding_quantile(unitvol,quantile)
    VarR=pd.rolling_quantile(unitvol,varwindow,quantile)
    
    normedquantile=Var*vol
    normedquantileR=VarR*vol
    
    ret2=ret.shift(-1)
    courbe=pd.DataFrame({'returns':ret2,
                  'quantiles':unnormedquantile,
                  'Rolling quantiles':unnormedquantileR,
                  'Normed quantiles':normedquantile,
                  'Rolling Normed quantiles':normedquantileR,
                  })
    
    courbe['nqBreak']=np.sign(ret2-normedquantile)/(-2) +0.5
    courbe['nqBreakR']=np.sign(ret2-normedquantileR)/(-2) +0.5
    courbe['UnqBreak']=np.sign(ret2-unnormedquantile)/(-2) +0.5
    courbe['UnqBreakR']=np.sign(ret2-unnormedquantileR)/(-2) +0.5
    
    nbdays=price.count()
    print('Number of returns worse than the VaR')
    print('Ideal Var                : ', (quantile)*nbdays)
    print('Simple VaR               : ', np.sum(courbe['UnqBreak']))
    print('Normalized VaR           : ', np.sum(courbe['nqBreak']))
    print('---------------------------')
    print('Ideal Rolling Var        : ', (quantile)*(nbdays-varwindow))
    print('Rolling VaR              : ', np.sum(courbe['UnqBreakR']))
    print('Rolling Normalized VaR   : ', np.sum(courbe['nqBreakR']))
    

if __name__ == "__main__":
    # print(bootstrap(0.048, 400, [(0.053, 91), (0.055, 98)]))
    # print(calcYieldToDate(95.0428, 100, 1.5, 5.75))
    # print(calcYieldToDate(100, 100, 2, 6))
    # xFound = newton_raphson(quadratic, 5, 0.01)    # call the solver
    # print("solution: x = ", xFound)
    # VaR()
    t_curve = [(datetime.date(2018,1,1), 0.021), (datetime.date(2019,1,1), 0.03635)]
    # crv = Curve(rates=[r[1] for r in t_curve], dts = [r[0] for r in t_curve])
    # calcZSprd(crv, 100.125, 100, datetime.date(2019,1,1), 0.0, freq=1,
    #             start_date=datetime.date(2017,1,1))