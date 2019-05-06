#
# Analyzing Euribor Interest Rate Data
# Source: http://www.emmi-benchmarks.eu/euribor-org/euribor-rates.html
# 03_stf/EURIBOR_analysis.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PNG_PATH = '../png/3ch/'

# Read Data for Euribor from Excel file


def read_euribor_data():
    ''' Reads historical Euribor data from Excel file, calculates log returns, 
    realized variance and volatility.'''
    EBO = pd.read_excel('../data/EURIBOR_current.xlsx',
                        index_col=0)
    EBO['returns'] = np.log(EBO['1w'] / EBO['1w'].shift(1))
    EBO = EBO.dropna()
    return EBO


# Plot the Term Structure
markers = [',', '-.', '--', '-']


def plot_term_structure(data):
    ''' Plot the term structure of Euribor rates. '''
    plt.figure(figsize=(10, 5))
    for i, mat in enumerate(['1w', '1m', '6m', '12m']):
        plt.plot(data[mat].index, data[mat].values,
                 'b%s' % markers[i], label=mat)
    plt.grid()
    plt.legend()
    plt.xlabel('strike')
    plt.ylabel('implied volatility')
    plt.ylim(0.0, plt.ylim()[1])
    plt.savefig(PNG_PATH + "plot_term_structure", dpi=300)
    plt.close()
    
    
if __name__ == '__main__':
    data = read_euribor_data()
    plot_term_structure(data)