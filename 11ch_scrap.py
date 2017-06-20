import sys, pdb
sys.path.append('/usr/share/doc')
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
np.random.seed(1000)
import scipy.stats as scs
import statsmodels.api as sm
import pandas as pd
from pandas_datareader import data as web
from utils import timeme

PATH = '/home/ubuntu/workspace/python_for_finance/png/ch11/'

def gen_paths(S0, r, sigma, T, M, I):
    ''' Generate Monte Carlo paths for geometric Brownian motion.
    
    Parameters
    ==========
    S0 : float
        initial stock/index value
    r : float
        constant short rate
    sigma : float
        constant volatility
    T : float
        final time horizon
    M : int
        number of time steps/intervals
    I : int
        number of paths to be simulated
        
    Returns
    =======
    paths : ndarray, shape (M + 1, I)
        simulated paths given the parameters
    '''
    dt = float(T) / M
    paths = np.zeros((M + 1, I), np.float64)
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * np.sqrt(dt) * rand)
    return paths

def normality_tests(arr):
    ''' Tests for normality distribution of given data set.
    
    Parameters
    ==========
    array: ndarray
        object to generate statistics on
    '''
    print("Skew of data set  %14.3f" % scs.skew(arr))
    print("Skew test p-value %14.3f" % scs.skewtest(arr)[1])
    print("Kurt of data set  %14.3f" % scs.kurtosis(arr))
    print("Kurt test p-value %14.3f" % scs.kurtosistest(arr)[1])
    print("Norm test p-value %14.3f" % scs.normaltest(arr)[1])

def print_statistics(array):
    ''' Prints selected statistics.
    
    Parameters
    ==========
    array: ndarray
        object to generate statistics on
    '''
    sta = scs.describe(array)
    print("%14s %15s" % ('statistic', 'value'))
    print(30 * "-")
    print("%14s %15.5f" % ('size', sta[0]))
    print("%14s %15.5f" % ('min', sta[1][0]))
    print("%14s %15.5f" % ('max', sta[1][1]))
    print("%14s %15.5f" % ('mean', sta[2]))
    print("%14s %15.5f" % ('std', np.sqrt(sta[3])))
    print("%14s %15.5f" % ('skew', sta[4]))
    print("%14s %15.5f" % ('kurtosis', sta[5]))

def norm_tests():
    S0 = 100.
    r = 0.05
    sigma = 0.2
    T = 1.0
    M = 50
    I = 250000
    paths = gen_paths(S0, r, sigma, T, M, I)

    plt.plot(paths[:, :10])
    plt.grid(True)
    plt.xlabel('time steps')
    plt.ylabel('index level')
    plt.savefig(PATH + 'norm_tests_paths.png', dpi=300)
    plt.close()

    log_returns = np.log(paths[1:] / paths[0:-1])
    print(paths[:, 0].round(4))
    print(log_returns[:, 0].round(4))

    print_statistics(log_returns.flatten())
    
    plt.hist(log_returns.flatten(), bins=70, normed=True, label='frequency')
    plt.grid(True)
    plt.xlabel('log-return')
    plt.ylabel('frequency')
    x = np.linspace(plt.axis()[0], plt.axis()[1])
    plt.plot(x, scs.norm.pdf(x, loc=r / M, scale=sigma / np.sqrt(M)),
             'r', lw=2.0, label='pdf')
    plt.legend()
    plt.savefig(PATH + 'norm_tests1.png', dpi=300)
    plt.close()
    
    sm.qqplot(log_returns.flatten()[::500], line='s')
    plt.grid(True)
    plt.xlabel('theoretical quantiles')
    plt.ylabel('sample quantiles')
    plt.savefig(PATH + 'norm_tests2.png', dpi=300)
    plt.close()
    
    normality_tests(log_returns.flatten())


    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.hist(paths[-1], bins=30)
    ax1.grid(True)
    ax1.set_xlabel('index level')
    ax1.set_ylabel('frequency')
    ax1.set_title('regular data')
    ax2.hist(np.log(paths[-1]), bins=30)
    ax2.grid(True)
    ax2.set_xlabel('log index level')
    ax2.set_title('log data')
    plt.savefig(PATH + 'norm_tests3.png', dpi=300)
    plt.close()

    print_statistics(paths[-1])
    print_statistics(np.log(paths[-1]))
    normality_tests(np.log(paths[-1]))

    log_data = np.log(paths[-1])
    plt.hist(log_data, bins=70, normed=True, label='observed')
    plt.grid(True)
    plt.xlabel('index levels')
    plt.ylabel('frequency')
    x = np.linspace(plt.axis()[0], plt.axis()[1])
    plt.plot(x, scs.norm.pdf(x, log_data.mean(), log_data.std()),
             'r', lw=2.0, label='pdf')
    plt.legend()
    plt.savefig(PATH + 'norm_tests4.png', dpi=300)
    plt.close()

    sm.qqplot(log_data, line='s')
    plt.grid(True)
    plt.xlabel('theoretical quantiles')
    plt.ylabel('sample quantiles')
    plt.savefig(PATH + 'norm_tests5.png', dpi=300)
    plt.close()
    
def real_world_data():
    # symbols = ['^GDAXI', '^GSPC', 'YHOO', 'MSFT']
    symbols = ['AAPL', 'IBM', 'YHOO', 'MSFT']
    data = pd.DataFrame()
    for sym in symbols:
        data[sym] = web.DataReader(sym, data_source='google', start='1/1/2006')['Close']
    
    data = data.dropna()
    print(data.info())
    print(data.head())
    (data / data.ix[0] * 100).plot(figsize=(8, 6), grid=True)
    plt.savefig(PATH + 'real_world_data1.png', dpi=300)
    plt.close()
    
    log_returns = np.log(data / data.shift(1))
    print(log_returns.head())
    log_returns.hist(bins=50, figsize=(9, 6))
    plt.savefig(PATH + 'real_world_data2.png', dpi=300)
    plt.close()

    for sym in symbols:
        print("\nResults for symbol %s" % sym)
        print(30 * "-")
        log_data = np.array(log_returns[sym].dropna())
        print_statistics(log_data)

    sm.qqplot(log_returns['AAPL'].dropna(), line='s')
    plt.grid(True)
    plt.xlabel('theoretical quantiles')
    plt.ylabel('sample quantiles')
    plt.savefig(PATH + 'real_world_data3.png', dpi=300)
    plt.close()

    sm.qqplot(log_returns['MSFT'].dropna(), line='s')
    plt.grid(True)
    plt.xlabel('theoretical quantiles')
    plt.ylabel('sample quantiles')
    plt.savefig(PATH + 'real_world_data4.png', dpi=300)
    plt.close()

    for sym in symbols:
        print("\nResults for symbol %s" % sym)
        print(32 * "-")
        log_data = np.array(log_returns[sym].dropna())
        normality_tests(log_data)


if __name__ == '__main__':
    # norm_tests()
    real_world_data()