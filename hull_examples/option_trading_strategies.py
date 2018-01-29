import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from dx import *
from utils.utils import *

import numpy as np
# import pandas as pd
import datetime as dt

PATH = '/home/ubuntu/workspace/python_for_finance/png/hull/options_trading/'

class option():
    '''represents a vanilla option'''
    def __init__(self, strike, prem, typ, mat_dt, short=False):
        self.k = strike
        self.typ = typ
        self.mat_dt = mat_dt
        self.short = short
        self.prem = prem
    
    def payoff(self, underlying):
        pay = 0
        if self.typ == 'c':
            pay = max(underlying - self.k, 0)
        else:
            pay = max(self.k - underlying, 0)
        
        if self.short:
            pay *= -1
            pay += self.prem
        else:
            pay -= self.prem
        return pay


def graph_payouts(opts, prices, filename, underlying=0):
    # underlying = -1 for short and 1 for long
    y = []
    for p in prices:
        y.append(0)
        for o in opts:
            if underlying:
                y[-1] += o.payoff(p) + underlying * (p - o.k)
            else:
                y[-1] += o.payoff(p)
    
    pdb.set_trace()
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.plot(prices, y, 'r')
    ax.axhline(y=0,color='b')
    plt.grid(True)
    ax.set_ylim([-40, 40])
    ax.set_xlim([0, prices[-1]])
    l1 = plt.legend(['payoff'], loc=2)
    plt.savefig(PATH + filename, dpi=300)
    plt.close()

if __name__ == '__main__':
    prices = np.linspace(0,110, 11)
    
    # option + underling
    # opts = [
    #     option(50, 10, 'c', dt.datetime(2015,1,1), False)
    # ]
    # graph_payouts(opts, prices, 'call_short_stock.png', -1)
    
    
    # opts = [
    #     option(40, 20, 'c', dt.datetime(2015,1,1), False),
    #     option(60, 10, 'c', dt.datetime(2015,1,1), True)
    # ]
    # graph_payouts(opts, prices, 'bull_spread.png')
    
    # opts = [
    #     option(40, 10, 'p', dt.datetime(2015,1,1), True),
    #     option(60, 20, 'p', dt.datetime(2015,1,1), False)
    # ]
    # graph_payouts(opts, prices, 'bear_spread.png')
    
    # opts = [
    #     option(40, 20, 'c', dt.datetime(2015,1,1), False),
    #     option(60, 10, 'c', dt.datetime(2015,1,1), True),
    #     option(40, 10, 'p', dt.datetime(2015,1,1), True),
    #     option(60, 20, 'p', dt.datetime(2015,1,1), False)
    # ]
    # graph_payouts(opts, prices, 'box_spread.png')
    
    # opts = [
    #     option(30, 30, 'c', dt.datetime(2015,1,1), False),
    #     option(70, 10, 'c', dt.datetime(2015,1,1), False),
    #     option(50, 19, 'c', dt.datetime(2015,1,1), True),
    #     option(50, 19, 'c', dt.datetime(2015,1,1), True)
    # ]
    # graph_payouts(opts, prices, 'butterfly_spread.png')
    
    # opts = [
    #     option(50, 10, 'c', dt.datetime(2015,1,1), False),
    #     option(50, 10, 'p', dt.datetime(2015,1,1), False)
    # ]
    # graph_payouts(opts, prices, 'straddle.png')
    
    opts = [
        option(60, 5, 'c', dt.datetime(2015,1,1), False),
        option(40, 5, 'p', dt.datetime(2015,1,1), False)
    ]
    graph_payouts(opts, prices, 'strangle.png')
    
    