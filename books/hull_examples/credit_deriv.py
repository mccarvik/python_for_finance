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


def cds_valuation(haz, T, rf, recov_rate, freq):
    # probability of surviving through a given year
    surv_rates = [np.exp(-haz*t) for t in range(T+1)]
    # probability of default in a given year
    def_rates = [surv_rates[t] - surv_rates[t+1] for t in range(T)]
    # discounted present value of an expected payment
    pv_exp_payments = sum([surv_rates[t] * np.exp(-rf*(t)) for t in range(T+1)]) - 1
    # assume default happens halway thru period
    pv_if_dflt = sum([def_rates[t] * np.exp(-rf*(t + freq/2)) * (1-recov_rate) for t in range(T)])
    # accrual payments --> if default in middle of period, investor still owed half of coupon
    pv_of_accrual = sum([def_rates[t] * freq/2 * np.exp(-rf * (t + freq/2)) for t in range(T)])
    cds_spread = pv_if_dflt / (pv_exp_payments + pv_of_accrual) * 100
    duration = pv_if_dflt / cds_spread
    return cds_spread, duration


def cds_price(haz, cds_sprd, cpn, recov_rate, freq, T, rf, n_companies_idx, dur, prot_per_comp):
    # credit spread = (Upfront premium/Duration) + Fixed coupon
    # should be able to calc duration and hazard rate from other information
    px = 100 - (100 * dur * (cds_sprd - cpn))
    pdb.set_trace()
    pay = prot_per_comp * n_companies_idx * ((px - 100) / 100)
    return pay


if __name__ == '__main__':
    # print(cds_valuation(0.02, 5, 0.05, 0.4, 1))
    print(cds_price(0.005717, 0.00345, 0.00406, 0.4, 0.25, 5, 0.04, 125, 4.447, 1))