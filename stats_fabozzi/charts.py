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


if __name__ == '__main__':
    data = pd.read_csv('dow.csv', header=None)
    # pie_chart(data)