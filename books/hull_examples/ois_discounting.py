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
yields = [0.01, 0.0366, 0.04]
dates = [dt.datetime(2015,1,1), dt.datetime(2016,7,1), dt.datetime(2017,1,1)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))

# present value of expected cost to bank of counterparty default
# same calc for debt value adjustment i.e. the value to the bank of its own default
# Value of the portfolio will be port value - CVA + DVA
# Also needs to be incorporate CRA (collateral rate adjustment) which is to account for posted collateral
def credit_value_adjustment(prob_default, pv_loss):
    return sum([q * v for q,v in zip(prob_default, pv_loss)])
    

if __name__ == '__main__':
    print(credit_value_adjustment([.1, .2], [10, 25]))