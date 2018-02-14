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
import scipy.stats as ss
from math import sqrt, pi, log, e
from hull_examples.bsm_model import *

# yields = [0.0025, 0.01, 0.015, 0.025]
# dates = [dt.datetime(2015, 1, 1), dt.datetime(2015, 7, 1), dt.datetime(2015, 12, 31), dt.datetime(2018, 12, 31)]
yields = [0.01, 0.0366, 0.04]
dates = [dt.datetime(2015,1,1), dt.datetime(2016,7,1), dt.datetime(2017,1,1)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))


def delta(S0, K, r, T, vol, div, otype='C'):
    # delta is equal to N(d1), represents how much the option price moves per move in underlying
    if otype == 'C':
        return ss.norm.cdf(d1(S0, K, r, T, vol, div))
    else:
        return ss.norm.cdf(d1(S0, K, r, T, vol, div)) - 1

if __name__ == '__main__':
    print(delta(42, 40, 0.1, 0.5, 0.2, 0.0, "C"))
    print(delta(42, 40, 0.1, 0.5, 0.2, 0.0, "P"))
    