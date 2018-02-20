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


def beta(dep_var, indep_var):
    # covariance(dependent var, independent var) / variance(independant var)
    cov = covariance(dep_var, indep_var)
    var = indep_var.var()
    return cov / var
    

def alpha(dep_var, indep_var):
    # left over value for movements not described by beta
    b = beta(dep_var, indep_var)
    return dep_var.mean() - b * indep_var.mean()


if __name__ == '__main__':
    data1 = pd.read_csv('aapl.csv')
    data1.columns = ['date','aapl']
    data2 = pd.read_csv('spy.csv')
    data2.columns = ['date','spy']
    print(beta(data1['aapl'], data2['spy']))
    print(alpha(data1['aapl'], data2['spy']))