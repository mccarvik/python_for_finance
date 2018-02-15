import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# from dx import *
# from utils.utils import *

import numpy as np
import pandas as pd
import datetime as dt

PATH = '/home/ubuntu/workspace/python_for_finance/png/stats_fabozzi/multivar/'

def setup_data(data1, data2):
    data = pd.merge(data1, data2, how='inner', on='date')
    data['appl_dret'] = data['aapl'].pct_change(1).round(3)
    data['spy_dret'] = data['spy'].pct_change(1).round(3)
    data['spy_pos'] = data.apply(lambda x: 0 if x['spy_dret'] < 0 else 1, axis=1)
    data['spy_neg'] = data.apply(lambda x: 1 if x['spy_dret'] < 0 else 0, axis=1)
    return data

def covariance(data1, data2):
    # measure of joint variation --> when both numbers move away from mean simultaneously, creates large values and vice versa
    mx = data1.mean()
    my = data2.mean()
    
    tot = 0
    for x, y in zip(data1, data2):
        tot += (x - mx) * (y - my)
    return tot / len(data1)

def correlation(data1, data2):
    # Pearson correlation coefficient. Will always between -1 and 1
    std1 = data1.std()
    std2 = data2.std()
    return covariance(data1, data2) / (std1 * std2)


if __name__ == '__main__':
    data1 = pd.read_csv('aapl.csv')
    data1.columns = ['date','aapl']
    data2 = pd.read_csv('spy.csv')
    data2.columns = ['date','spy']
    data = setup_data(data1, data2)
    
    # need this for return bucketing
    data_rets = data.groupby('appl_dret')[['spy_pos', 'spy_neg']].sum()
    
    # Compare reutrns when negative or positive
    # print(data[data.spy_neg==1]['appl_dret'].mean())
    # print(data[data.spy_pos==1]['appl_dret'].mean())
    
    # covariance and correlation
    print(covariance(data1['aapl'], data2['spy']))
    print(correlation(data1['aapl'], data2['spy']))
    