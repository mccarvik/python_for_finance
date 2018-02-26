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

class distribution():
    def __init__(self, vals, weights):
        self.weights = weights
        self.vals = vals


def mean(dist):
    tot_val = 0
    for w, v in zip(dist.vals, dist.weights):
        tot_val += w * v
    return tot_val
    

def dice_dist():
    dist = distribution([], [])
    for i in range(1, 7):
        dist.vals.append(i)
        dist.weights.append(1/6)
    return dist



if __name__ == '__main__':
    print(mean(dice_dist()))