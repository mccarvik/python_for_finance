import sys, pdb
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import datetime as dt

from dx import *


def risk_neutral_discounting():
    dates = [dt.datetime(2015, 1, 1), dt.datetime(2015, 7, 1), dt.datetime(2016, 1, 1)]
    deltas = [0.0, 0.5, 1.0]
    csr = constant_short_rate('csr', 0.05)
    print(csr.get_discount_factors(dates))
    deltas = get_year_deltas(dates)
    print(deltas)
    print(csr.get_discount_factors(deltas, dtobjects=False))

def mkt_env():
    dates = [dt.datetime(2015, 1, 1), dt.datetime(2015, 7, 1), dt.datetime(2016, 1, 1)]
    deltas = [0.0, 0.5, 1.0]
    csr = constant_short_rate('csr', 0.05)
    
    me_1 = market_environment('me_1', dt.datetime(2015,1,1))
    me_1.add_list('symbols', ['AAPL', 'MSFT', 'FB'])
    print(me_1.get_list('symbols'))
    
    me_2 = market_environment('me_2', dt.datetime(2015,1,1))
    me_2.add_constant('volatility', 0.2)
    me_2.add_curve('short_rate', csr)
    print(me_2.get_curve('short_rate'))
    
    me_1.add_environment(me_2)
    print(me_1.get_curve('short_rate'))
    print(me_1.constants)
    print(me_1.lists)
    print(me_1.curves)
    print(me_1.get_curve('short_rate').short_rate)
    
if __name__ == '__main__':
    risk_neutral_discounting()
    mkt_env()