import os, pdb, time, sys, time
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'serif'

PNG_PATH = '../png/2ch/'

def inner_value():
    # Option Strike
    K = 8000
    
    # Graphical Output
    S = np.linspace(7000, 9000, 100)  # index level values
    h = np.maximum(S - K, 0)  # inner values of call option
    
    # Hockey stick graphs of option value
    plt.figure()
    plt.plot(S, h, lw=2.5)  # plot inner values at maturity
    plt.xlabel('index level $S_t$ at maturity')
    plt.ylabel('inner value of European call option')
    plt.grid(True)
    plt.savefig(PNG_PATH + "inner_value", dpi=300)
    plt.close()
    
    
if __name__ == '__main__':
    inner_value()