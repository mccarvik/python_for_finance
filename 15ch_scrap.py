import seaborn as sns; sns.set()
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
import numpy as np
import pandas as pd
import datetime as dt
import sys
sys.path.append('../python3/dxa')
from dx_frame import *

def risk_neutral_disc():
    dates = [dt.datetime(2015, 1, 1), dt.datetime(2015, 7, 1), dt.datetime(2016, 1, 1)]
    deltas = [0.0, 0.5, 1.0]
    csr = constant_short_rate('csr', 0.05)
    csr.get_discount_factors(dates)
    print(array([[datetime.datetime(2015, 1, 1, 0, 0), 0.951229424500714],
            [datetime.datetime(2015, 7, 1, 0, 0), 0.9755103387657228],
            [datetime.datetime(2016, 1, 1, 0, 0), 1.0]], dtype=object))
    deltas = get_year_deltas(dates)
    print(deltas)
    print(csr.get_discount_factors(deltas, dtobjects=False))


if __name__ == '__main__':
    risk_neutral_disc()