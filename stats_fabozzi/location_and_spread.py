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
    
def quantiles(data, quantile):
    # Will return whatever percentil is input in the data
    # Ex: 75 will return the data point with 75% below it and 25% above it
    return np.percentile(data, quantile)

# Returns min, max, dist between min and max
def sample_range(data):
    return (data.min(), data.max(), data.max() - data.min())
    
# Returns bottom percentile, top percentile, and dist between, default to 25% and 75%
def interquartile_range(data, bottom=25, top=75):
    return (np.percentile(data, bottom), np.percentile(data, top), np.percentile(data, top) - np.percentile(data, bottom))

# Returns average absolute deviation from mean or median
def absolute_deviation(data, med=True):
    tot = 0
    middle = None
    if med:
        middle = median(data)
    else:
        middle = mean(data)
    
    for i in data:
        tot += abs(i - middle)
    return tot / len(data)
    
def variance(data):
    m = mean(data)
    tot = 0
    for i in data:
        tot += (m-i)**2
    return tot / len(data)

# approximate average deviation from the mean
def standard_dev(data):
    return variance(data)**(0.5)


# Pearson Skewness
def skewness(data):
    # mean > median --> skewed right, positive skew
    # median > mean --> skewed left, negative skew
    return ((mean(data) - median(data)) / standard_dev(data))

# Another type of skewness involving third power of mean and standard deviation
def skewness_third(data):
    tot = 0
    m = mean(data)
    stdev = standard_dev(data)
    for i in data:
        tot += (i - m)**3
    return (tot / (len(data) * stdev**3))

# needed to compare st_dev between different samples
def coeff_variation(data):
    return standard_dev(data) / mean(data)
    
def standardize_data(data):
    ret = []
    m = mean(data)
    st_dev = standard_dev(data)
    for i in data:
        ret.append((i - m) / st_dev)
    return ret

if __name__ == '__main__':
    data = pd.read_csv('dow.csv', header=None)
    # print(mean(data[1]))
    # print(median(data[1]))
    # print(median_class([8,15], 0.349, 0.630))
    # print(mode([1,2,3,3,4,5,1,2,2,1,1]))
    # wgts = np.linspace(0, 1, 30)
    # print(weighted_mean(data[1], wgts))
    # print(quantiles(data[1], 75))
    # print(sample_range(data[1]))
    # print(interquartile_range(data[1]))
    # print(absolute_deviation(data[1]))
    # print(absolute_deviation(data[1], med=False))
    # print(variance(data[1]))
    # print(standard_dev(data[1]))
    
    # print(skewness(data[1]))
    # print(skewness_third(data[1]))
    print(coeff_variation(data[1]))
    print(standardize_data(data[1]))