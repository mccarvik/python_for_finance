import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from math import sqrt, pi, log, e

# from dx import *
# from utils.utils import *

import numpy as np
import pandas as pd
import datetime as dt

from continuous_prob_dists import normal_dist, students_t_dist, chi_squared_dist


def conf_intervals(a, d='normal'):
    if d == 'normal':
        dist = normal_dist(0, 1)
    if d == 'tsutdent':
        dist = students_t_dist(3)
    if d == 'chi':
        dist = chi_squared_dist(5)
    
    # interval = dist.conf_interval(a)
    # interval = dist.conf_interval(a, mv=0.01, f=4, b=-4)
    interval = dist.conf_interval(a, mv=0.01, f=14, b=0)
    return interval
    

if __name__ == '__main__':
    # print(conf_intervals(0.05))
    print(conf_intervals(0.05, d='tsutdent'))
    print(conf_intervals(0.05, d='chi'))