import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

# import numpy as np
# import pandas as pd
import datetime as dt

from dx import *

yields = [0.0025, 0.01, 0.015]
dates = [dt.datetime(2015, 1, 1), dt.datetime(2015, 7, 1), dt.datetime(2015, 12, 31)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))

# returns the value of a money market deposit
def money_market_deposit(settle_dt, mat_dt):
    delta_t = get_year_deltas([settle_dt, mat_dt])[-1]
    yld = dsr.get_interpolated_yields([settle_dt, mat_dt])[-1][1]
    V = 1 + delta_t * yld
    return V

if __name__ == '__main__':
    print(money_market_deposit(dt.datetime(2015, 1, 1), dt.datetime(2015, 5, 1)))