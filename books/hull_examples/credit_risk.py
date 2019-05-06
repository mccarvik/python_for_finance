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
import pandas as pd
import datetime as dt
from functools import reduce
from math import sqrt, pi, log, e
import scipy.stats as ss
from bsm_model import *
from greeks import N

N_inv = ss.norm.ppf

def calc_hazard_rates(rec_rate, yld_sprds):
    # yld_spreads are in basis points
    
    # calc hazard rates from t0 to year x
    # ex: first rate is hazard rate for yr 1, second hazard rate is for yrs 1 and 2, ... etc
    cum_haz = []
    for y in yld_sprds:
        cum_haz.append((y / (1 - rec_rate))/10000)
    
    # calc hazard rates for each individual year
    # ex: first rate is hazard rate for yr 1, second hazard rate for just year 2, third just for year 3, etc
    ind_haz = []
    for i in range(len(yld_sprds)):
        if i == 0:
            continue
        ind_haz.append((i+1) * cum_haz[i] - (i * cum_haz[i-1]))
    
    return (cum_haz, ind_haz)


def match_bond_prices(yld_dt, rf, recov_rate=0.40, par=100, freq=0.5, start_dt=dt.datetime(2017, 1, 1)):
    hazard_rates = []
    prob_of_default = []
    for i in yld_dt:
        cfs = createCashFlows(start_dt, 0.5, dt.datetime(2017+i[0], 1, 1), 0.08, par)
        rf_pv = cumPresentValue(start_dt, rf, cfs)
        pv = cumPresentValue(start_dt, i[1], cfs)
        pv_exp_default = rf_pv - pv
        pv_loss = 0
        pv_losses = []
        t = 0
        while t < len(cfs)-1:
            t_cfs = cfs[t:]
            for cf in t_cfs:
                frac_date = get_year_deltas([start_dt, cf[0]])[-1]
                # assume middle of period
                frac_date = frac_date - freq / 2 - (freq*t)
                pv_loss += cf[1] * np.exp(-0.05*frac_date)
            frac_date = get_year_deltas([start_dt, t_cfs[0][0]])[-1] - freq / 2
            pv_losses.append((pv_loss - recov_rate*par) * np.exp(-0.05*frac_date))
            pv_loss = 0
            t += 1
        
        first_legs = [p_of_d * pv for p_of_d, pv in zip(prob_of_default, pv_losses)]
        pv_losses = pv_losses[len(first_legs):]
        # func = lambda x: ((1 - np.exp(-0.5 * x)) * pv_losses[0] + ((1 - np.exp(-0.5 * x)**2) - (1 - np.exp(-0.5 * x))) * pv_losses[1]) + sum(first_legs) - pv_exp_default
        func = lambda x: sum([(np.exp(-0.5 * i * x) - np.exp(-0.5 * (i+1) * x)) * pv_losses[i] for i in range(len(pv_losses))]) + sum(first_legs) - pv_exp_default
        hazard_rates.append(newton_raphson(func, 0.03))
        prob_of_default += [(np.exp(-0.5 * i * hazard_rates[-1]) - np.exp(-0.5 * (i+1) * hazard_rates[-1])) for i in range(len(pv_losses))]
        # surv_rate = surv_rate of last year (or 1 if its first year) - prob_of_default
        print()
        
    return hazard_rates


def equity_as_call_on_assets(E, vol_e, r, T, D):
    # E = V0 * N(d1) - De**(-r * T) * N(d2)
    # vol_e * E = N(d1) * vol_v * V0
    # solving for the 2 equations give v0 = 12.40 and vol_v = 0.2123
    # very complicated to do programatically
    V0 = 12.40
    vol_v = 0.2123
    
    prob_of_default = N(-d2(V0, D, r, T, vol_v, 0))
    print(prob_of_default)
    mv_of_debt = V0 - E
    pv_of_debt = D * np.exp(-r*T)
    # expected loss as a percentage of no default value
    exp_loss_on_debt = (pv_of_debt - mv_of_debt) / pv_of_debt
    recovery_rate = (1 - exp_loss_on_debt / prob_of_default)
    print(recovery_rate)
    return recovery_rate
    
    
def credit_mitigation(px_opt, t, bond_yld):
    # need to increase the discount rate on a derivative payoff to factor in that it might not get paid off
    # discount the value of the derivative by the yld over the risk free rate of the counterparty's debt
    new_val = px_opt * np.exp(-bond_yld * t)
    return new_val


def credit_risk_mitigiation_fwd(K, T, F, dflts, r, recov_rate, vol_g):
    # T = array of time periods where default is possible
    d_1 = d1(F, K, 0, T[0]/2, vol_g, 0)
    d_2 = d2(F, K, 0, T[0]/2, vol_g, 0)
    wt1 = np.exp(-r*((T[1]-T[0])/2 + T[0])) * (F * N(d_1) - K * N(d_2))
    vt1 = wt1 * np.exp(-r*T[0]/2) * (1 - recov_rate)
    
    d_1 = d1(F, K, 0, (T[1]-T[0])/2 + T[0], vol_g, 0)
    d_2 = d2(F, K, 0, (T[1]-T[0])/2 + T[0], vol_g, 0)
    wt2 = np.exp(-r*(T[0]/2)) * (F * N(d_1) - K * N(d_2))
    vt2 = wt2 * np.exp(-r*((T[1]-T[0])/2 + T[0])) * (1 - recov_rate)
    
    # The cost of default at each stage
    exp_cost_of_default = dflts[0] * vt1 + dflts[1] * vt2
    no_def_val = (F - K) * np.exp(-r*T[1])
    val_with_defaults = no_def_val - exp_cost_of_default
    return val_with_defaults


def credit_var(A, conf, dflt, corr, recov_rate):
    num = N_inv(0.02) + np.sqrt(corr) * N_inv(conf)
    denom = np.sqrt(1 - corr)
    worst_case_dflt_rate = N(num / denom)
    return A * worst_case_dflt_rate * (1 - recov_rate)
    
    


if __name__ == '__main__':
    # print(calc_hazard_rates(0.40, [150, 180, 195]))
    # yld_dt_pairs = [[1, 0.065], [2, 0.068], [3, 0.0695]]
    # print(match_bond_prices(yld_dt_pairs, 0.05))
    # print(equity_as_call_on_assets(3, 0.80, 0.05, 1, 10))
    # print(credit_mitigation(3, 2, 0.015))
    # print(credit_risk_mitigiation_fwd(1500, [1, 2], 1600, [0.02, 0.03], 0.05, 0.3, 0.2))
    print(credit_var(100, 0.999, 0.02, 0.1, 0.6))
    