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


yields = [0.01, 0.05, 0.04, 0.01]
dates = [dt.datetime(2015,1,1), dt.datetime(2015,4,1), dt.datetime(2015,10,1), dt.datetime(2017, 1, 1)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))


def eurodollar_futures_price(quote):
    # 25$ per basis point move
    return 10000 * (100 - 0.25 *(100 - quote))


# Used to account for the difference between the forward and futures rate
def convexity_adjustment(fut_rat, sigma, settle_dt, T1, T2):
    # sigma = st_dev of short term interest rate in 1 year
    T1 = get_year_deltas([settle_dt, T1])[-1]
    T2 = get_year_deltas([settle_dt, T2])[-1]
    return fut_rat - (0.5) * sigma**2 * T1 * T2


if __name__ == "__main__":
    print(eurodollar_futures_price(99.725))
    print(convexity_adjustment(0.06038, 0.012, dt.datetime(2015,1,1), dt.datetime(2023,1,1), dt.datetime(2023,4,1)))