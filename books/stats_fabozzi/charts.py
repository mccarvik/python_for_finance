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

PATH = '/home/ubuntu/workspace/python_for_finance/png/stats_fabozzi/charts/'


def pie_chart(data):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = data[0][:5]
    sizes = data[1][:5]
    # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice 

    fig1, ax1 = plt.subplots()
    # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(PATH + 'pie_chart.png', dpi=300)
    plt.close()


def bar_chart(data):
    fig, ax = plt.subplots()
    pdb.set_trace()
    ax.bar(range(len(data)), data[1], 0.35, color='r')
    ax.set_xticklabels(data[0])
    plt.savefig(PATH + 'bar_chart.png', dpi=300)
    plt.close()


def histogram(data, ogive=True):
    fig, ax = plt.subplots()
    counts, bins, patches = plt.hist(data, 8)
    centers =[]
    if ogive:
        for b in range(len(bins)):
            if b==0:
                continue
            try:
                centers.append((bins[b] + bins[b+1])/2)
            except:
                centers.append(bins[b])
        ax.plot(centers, counts.cumsum(), 'ro-')
    
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(PATH + 'histogram.png', dpi=300)
    plt.close()


def box_plot(data):
    # change outlier point symbols
    plt.figure()
    plt.boxplot(data, 0, 'gD')
    plt.savefig(PATH + 'boxplot.png', dpi=300)
    plt.close()


def qq_plot(data_1, data_2):
    # compares two sets of data against eachother and sees how the relationship behaves
    pdb.set_trace()
    plt.scatter(data_1, data_2, c="g")
    plt.savefig(PATH + 'qqplot.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    data = pd.read_csv('aapl.csv')
    # pie_chart(data)
    # bar_chart(data)
    # histogram(data[1])
    # box_plot(data[1])
    pdb.set_trace()
    qq_plot(data['Close'][:100], data['Close'][-100:])