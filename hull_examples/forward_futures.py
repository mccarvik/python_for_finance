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


yields = [0.01, 0.10, 0.04]
dates = [dt.datetime(2015,1,1), dt.datetime(2015,7,1), dt.datetime(2015,10,1)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))


def forward_cont_disc(price, settle_dt, mat_dt, income_pv=0, yld=0):
    # income_pv is the present value of income from the underlying asset
    t = get_year_deltas([settle_dt, mat_dt])[-1]
    r = dsr.get_interpolated_yields([settle_dt, mat_dt])[1][1]
    if not yld:
        return (price - income_pv) * np.exp(r * t)
    else:
        return (price) * np.exp((r - yld) * t)


if __name__ == '__main__':
    # print(forward_cont_disc(900, dt.datetime(2015,1,1), dt.datetime(2015,10,1), income_pv=39.60))
    print(forward_cont_disc(25, dt.datetime(2015,1,1), dt.datetime(2015,7,1), yld=0.0396))