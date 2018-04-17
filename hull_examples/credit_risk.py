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
from math import sqrt, pi, log, e


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



if __name__ == '__main__':
    print(calc_hazard_rates(0.40, [150, 180, 195]))