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


# NOTES
# multicollinearity - correlation between the independent variables and the dependent variable
# Variance Inflation Factor (VIF) = (1 / (1 - R**2)) --> if no correlation between variables than variance will remain
# the same even after addition of new variable
# dummy variables - either 0 or 1 based on the existence of some trait, ex: member of energy sector


def f_stat(data):
    # k = num of vars
    n = len(vals)
    MSR = SSR(vals) / k
    MSE = SSE / (n - k - 1)
    return MSR / MSE


def setup_data(y, xs):
    # only works for 2 vairables for now
    data = pd.merge(y, xs[0], how='inner', on='date')
    data = pd.merge(data, xs[1], how='inner', on='date')
    return data


def F_test(data):
    # Measures if a new variable to the model will increase the value of the model
    # R's = R**2 values of goodness of fit
    num = (R1 - R2)
    denom = (1 - R2) / (n - k - 1)
    return num / denom


if __name__ == '__main__':
    data1 = pd.read_csv('aapl.csv')
    data2 = pd.read_csv('amzn.csv')
    data3 = pd.read_csv('spy.csv')
    data1.columns = ['date','aapl']
    data2.columns = ['date','amzn']
    data3.columns = ['date','spy']
    data = setup_data(data3, [data1, data2])
    f_stat(data)