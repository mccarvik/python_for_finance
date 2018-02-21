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


def SST(dep_var):
    # Total Sum of Squares --> variance of the observations
    # SST = SSR + SSE
    m = dep_var.mean()
    return sum([(y - m)**2 for y in dep_var])


def SSR(dep_var, indep_var):
    # Sum of Squares explained by the Regression
    # SSR = Sum((y_predicted - y_mean)^2)
    m = dep_var.mean()
    b = beta(dep_var, indep_var)
    a = alpha(dep_var, indep_var)
    return sum([((x*b + a) - m)**2 for x in indep_var])


def SSE(dep_var, indep_var):
    # Sum of Squared Errors
    # SSE = Sum((y - y_predicted)^2)
    b = beta(dep_var, indep_var)
    a = alpha(dep_var, indep_var)
    return sum([(y - (x*b + a))**2 for x, y in zip(indep_var, dep_var)])


def r_squared(dep_var, indep_var):
    # Coefficient of Determination
    # value from 0 to 1, higher value = better fit
    return SSR(dep_var, indep_var) / SST(dep_var)
    # return 1 - SSE(dep_var, indep_var) / SST(dep_var)


if __name__ == '__main__':
    data1 = pd.read_csv('aapl.csv')
    data1.columns = ['date','aapl']
    data2 = pd.read_csv('spy.csv')
    data2.columns = ['date','spy']
    # print(beta(data1['aapl'], data2['spy']))
    # print(alpha(data1['aapl'], data2['spy']))
    
    print(SST(data1['aapl']))
    print(SSR(data1['aapl'], data2['spy']))
    print(SSE(data1['aapl'], data2['spy']))
    print(r_squared(data1['aapl'], data2['spy']))
    