import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

from dx import *
from utils.utils import *

import numpy as np
# import pandas as pd
import datetime as dt


def price_binary_tree(underlying, vol, strike, r, settle_dt, mat_dt, otype='C', N=2000, typ='euro'):
    # N = number of steps of tree
    S0 = underlying
    sigma = vol
    K = strike
    T = get_year_deltas([settle_dt, mat_dt])[-1]
    
    #calculate delta T    
    deltaT = float(T) / N
 
    # up and down factor will be constant for the tree so we calculate outside the loop
    # u, d = the up, down % move per step
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1.0 / u
    u = 1.2
    d= 0.8
    deltaT = 1
 
   
    # Initialise our f_{i,j} tree with zeros
    fs = [[0.0 for j in range(i + 1)] for i in range(N + 1)]
    
    #store the tree in a triangular matrix - this is the closest to theory
    #no need for the stock tree
 
    #rates are fixed so the probability of up and down are fixed.
    #this is used to make sure the drift is the risk free rate
    # p = probability of an up move
    a = np.exp(r * deltaT)
    p = (a - d) / (u - d)
    oneMinusP = 1.0 - p
    
    # Compute the leaves, f_{N, j}
    for j in range(len(fs)):
        # j = number of up moves at this leaf
        if otype =="C":
            fs[N][j] = max(S0 * u**j * d**(N - j) - K, 0.0)
        else:
            fs[N][j] = max(-S0 * u**j * d**(N - j) + K, 0.0)
            
    #calculate backward the option prices
    for i in range(N-1, -1, -1):
        for j in range(i + 1):
            # Adjust for potential early exercise
            if typ == 'amer':
                if otype =="C":
                    early_ex = max(S0 * u**(j) * d**(i - j) - K, 0.0)
                else:
                    early_ex = max(-S0 * u**(j) * d**(i - j) + K, 0.0)
            else:
                early_ex = 0
            
            # The PV of the up node + the PV of the down node, discount for continuous compounding TVM
            fs[i][j] = max(np.exp(-r * deltaT) * (p * fs[i + 1][j + 1] + oneMinusP * fs[i + 1][j]), early_ex)
    return fs[0][0]


if __name__ == '__main__':
    print(price_binary_tree(50, 0.2, 52, 0.05, dt.datetime(2015,1,1), dt.datetime(2016,1,1), otype='P', N=2, typ='amer'))