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

PATH = '/home/ubuntu/workspace/python_for_finance/png/stats_fabozzi/'

def mean(data):
    tot = 0
    for i in data:
        tot += i
    return tot / len(data)

def weighted_mean(data, wgts):
    tot = 0
    for i, w in zip(data, wgts):
        tot += i * w
    return tot / sum(wgts)

def median(data):
    srt = sorted(data)
    if len(srt) % 2 == 1:
        return srt[int((len(srt)-1) / 2)]
    else:
        return (srt[int(len(srt) / 2) - 1] +  srt[int(len(srt) / 2)]) / 2

def median_class(class_of_incident, bottom_cum_freq, top_cum_freq):
    # data is classifie - need a different process
    # find the class of incident = class where median is between top and bottom range
    # Linearly interpolate over the range of the class
    
    # range
    rng = class_of_incident[1] - class_of_incident[0]
    start = class_of_incident[0]
    linear_interpt = (0.5 -  bottom_cum_freq) / (top_cum_freq -  bottom_cum_freq) 
    return linear_interpt * rng + start
    
def mode(data):
    tots = {}
    mx = {'' : 0}
    for i in data:
        if i in list(tots.keys()):
            tots[i] += 1
        else:
            tots[i] = 1
        if tots[i] > mx[list(mx.keys())[0]]:
            mx = {i : tots[i]}
    return mx
    

if __name__ == '__main__':
    data = pd.read_csv('dow.csv', header=None)
    # print(mean(data[1]))
    # print(median(data[1]))
    # print(median_class([8,15], 0.349, 0.630))
    # print(mode([1,2,3,3,4,5,1,2,2,1,1]))
    wgts = np.linspace(0, 1, 30)
    print(weighted_mean(data[1], wgts))