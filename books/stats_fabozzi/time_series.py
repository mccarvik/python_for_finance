import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from stats_fabozzi.multivariable import covariance

# from dx import *
# from utils.utils import *

import numpy as np
import pandas as pd
import datetime as dt

def setup_data(data1, data2):
    data = pd.merge(data1, data2, how='inner', on='date')
    data['appl_dret'] = data['aapl'].pct_change(1).round(3)
    data['spy_dret'] = data['spy'].pct_change(1).round(3)
    data['spy_pos'] = data.apply(lambda x: 0 if x['spy_dret'] < 0 else 1, axis=1)
    data['spy_neg'] = data.apply(lambda x: 1 if x['spy_dret'] < 0 else 0, axis=1)
    return data


def decomp_of_ts(data):
    # x = trend + cyclical term + seasonal term + disturbance(or error) --> x = T + Z + S + U
    # combining seasonal and distrubance, I = (fade, 0 <f<1) * I(t-1) * U -- > x = T + S + I
    # Function returns average return per weekday over period to simulate an expample of seasonal term
    day_rets = {
        1 : [], 2 : [], 3 : [], 4 : [], 0 : []
    }
    for ix, row in data.iterrows():
        day = dt.datetime.strptime(row['date'], '%d-%b-%y').weekday()
        day_rets[day].append(row['spy_dret'])
    
    rets = []
    for d in day_rets:
        day_rets[d] = [d for d in day_rets[d] if not np.isnan(d)]
        rets.append(sum(day_rets[d]) / len(day_rets[d]) * 100)
    return rets


def random_walk(t=100, init_val=100, mv=1):
    for i in range(t):
        init_val+= (2 * np.random.rand() - 1) * mv
        print(init_val)
    return init_val


def error_correction(t=100, init_val=100, mv=1):
    # very similar to mean reverting where anytime the equilibirum is off it tries to get back
    F = init_val
    St1 = init_val
    err = (2 * np.random.rand() - 1) * mv
    # alpha > 0 and beta > 0
    alpha = 0.1
    beta = 0.1
    for i in range(t):
        err = (2 * np.random.rand() - 1) * mv
        St2 = St1 - alpha * (St1 - F) + err
        err = (2 * np.random.rand() - 1) * mv
        F += beta * (St1 - F) + err
        St1 = St2
        print(St1)
    return St1


if __name__ == '__main__':
    data1 = pd.read_csv('aapl.csv')
    data1.columns = ['date','aapl']
    data2 = pd.read_csv('spy.csv')
    data2.columns = ['date','spy']
    data = setup_data(data1, data2)
    # print(decomp_of_ts(data))
    # print(random_walk())
    print(error_correction())