import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
# sys.path.append("/usr/local/lib/python3.4/dist-packages")
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

# from dx import *
# from utils.utils import *

import numpy as np
import pandas as pd
import datetime as dt


def cum_freq_dist(data, x=50):
    tot = 0
    for ix, row in data.iterrows():
        if float(row[1]) > x:
            tot += 1
    return tot / len(data)
    

# Rule for the optimal number of classes for a set of data
def sturges_rule(data):
    # return 1 + np.log(len(data))
    return int((1 + 3.222 * np.log10(len(data))) + 0.5)
    

# Rule for the optimal width of classes for a set of data
def freedman_diaconis_rule(data):
    pdb.set_trace()
    q75, q25 = np.percentile(data[1], [75 ,25])
    iqr = q75 - q25
    return int((2 * iqr * len(data)**(-1/3)) + 0.5)


if __name__ == '__main__':
    data = pd.read_csv('dow.csv', header=None)
    # print(cum_freq_dist(data, 50))
    # print(sturges_rule(data))
    print(freedman_diaconis_rule(data))