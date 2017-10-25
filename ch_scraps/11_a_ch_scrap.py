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
import scipy.optimize as sco
import scipy.interpolate as sci
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
    I = 250
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

def min_func_sharpe(weights):
    return -statistics(weights)[2]

def real_world_data():
    # symbols = ['^GDAXI', '^GSPC', 'YHOO', 'MSFT']
    symbols = ['AAPL', 'IBM', 'YHOO', 'MSFT']
    data = pd.DataFrame()
    for sym in symbols:
        data[sym] = web.DataReader(sym, data_source='google', start='1/1/2006')['Close']
    
    data = data.dropna()
    print(data.info())
    print(data.head())
    (data / data.ix[0] * 100).plot(figsize=(7, 4), grid=True)
    plt.savefig(PATH + 'real_world_data1.png', dpi=300)
    plt.close()
    
    log_returns = np.log(data / data.shift(1))
    print(log_returns.head())
    log_returns.hist(bins=50, figsize=(7, 4))
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

def statistics(weights):
    ''' Return portfolio statistics.
    
    Parameters
    ==========
    weights : array-like
        weights for different securities in portfolio
    
    Returns
    =======
    pret : float
        expected portfolio return
    pvol : float
        expected portfolio volatility
    pret / pvol : float
        Sharpe ratio for rf=0
    '''
    rets = port_opt_data()
    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])

def min_func_variance(weights):
    rets = port_opt_data()
    return statistics(weights)[1] ** 2

def min_func_port(weights):
    return statistics(weights)[1]

def port_opt_data():
    # symbols = ['AAPL', 'MSFT', 'YHOO', 'DB', 'GLD']
    symbols = ['IBM', 'DB', 'GLD']
    noa = len(symbols)
    data = pd.DataFrame()
    for sym in symbols:
        data[sym] = web.DataReader(sym, data_source='google', start='2014-09-01', end='2014-09-12')['Close']
    data.columns = symbols
    (data / data.ix[0] * 100).plot(figsize=(8, 5), grid=True)
    rets = np.log(data / data.shift(1))
    return rets

def port_opt_theory():
    pdb.set_trace()
    rets = port_opt_data()
    noa = len(rets.columns)
    print(rets.mean() * 252)
    print(rets.cov() * 252)

    weights = np.random.random(noa)
    weights /= np.sum(weights)
    print(weights)
    print(np.sum(rets.mean() * weights) * 252)
    # expected portfolio variance
    print(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    # expected portfolio standard deviation/volatility 
    print(np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))))
    prets = []
    pvols = []
    for p in range (250):
        weights = np.random.random(noa)
        weights /= np.sum(weights)
        prets.append(np.sum(rets.mean() * weights) * 252)
        pvols.append(np.sqrt(np.dot(weights.T, 
                            np.dot(rets.cov() * 252, weights))))
    prets = np.array(prets)
    pvols = np.array(pvols)
    
    plt.figure(figsize=(8, 4))
    plt.scatter(pvols, prets, c=prets / pvols, marker='o')
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    # plt.colorbar(label='Sharpe ratio')
    plt.savefig(PATH + 'port_opt1.png', dpi=300)
    plt.close()
    
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    bnds = tuple((0, 1) for x in range(noa))
    print(noa * [1. / noa,])
    opts = timeme(sco.minimize)(min_func_sharpe, noa * [1. / noa,], method='SLSQP',
                           bounds=bnds, constraints=cons)
    print(opts)
    print(opts['x'].round(3))
    print(statistics(opts['x']).round(3))

    optv = timeme(sco.minimize)(min_func_variance, noa * [1. / noa,], method='SLSQP',
                           bounds=bnds, constraints=cons)
    print(optv)
    print(optv['x'].round(3))
    print(statistics(optv['x']).round(3))
    cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
            {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    bnds = tuple((0, 1) for x in weights)
    
    # Efficient Frontier
    # trets = np.linspace(0.0, 0.25, 50)
    trets = np.linspace(0.0, 0.25, 4)
    tvols = []
    for tret in trets:
        cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
                {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        res = timeme(sco.minimize)(min_func_port, noa * [1. / noa,], method='SLSQP',
                           bounds=bnds, constraints=cons)
        tvols.append(res['fun'])
    
    tvols = np.array(tvols)
    plt.figure(figsize=(8, 4))
    plt.scatter(pvols, prets,
                c=prets / pvols, marker='o')
                # random portfolio composition
    plt.scatter(tvols, trets,
                c=trets / tvols, marker='x')
                # efficient frontier
    plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
            'r*', markersize=15.0)
            # portfolio with highest Sharpe ratio
    plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
         'y*', markersize=15.0)
            # minimum variance portfolio
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    # plt.colorbar(label='Sharpe ratio')
    plt.savefig(PATH + 'port_opt_efficient_frontier.png', dpi=300)
    plt.close()
    
    ind = np.argmin(tvols)
    evols = tvols[ind:]
    erets = trets[ind:]
    tck = sci.splrep(evols, erets)
    
    def f(x):
        ''' Efficient frontier function (splines approximation). '''
        return sci.splev(x, tck, der=0)
    def df(x):
        ''' First derivative of efficient frontier function. '''
        return sci.splev(x, tck, der=1)

    def equations(p, rf=0.01):
        eq1 = rf - p[0]
        eq2 = rf + p[1] * p[2] - f(p[2])
        eq3 = p[1] - df(p[2])
        return eq1, eq2, eq3

    opt = sco.fsolve(equations, [0.01, 0.5, 0.15])
    print(opt)
    print(np.round(equations(opt), 6))

    plt.figure(figsize=(8, 4))
    plt.scatter(pvols, prets,
                c=(prets - 0.01) / pvols, marker='o')
                # random portfolio composition
    plt.plot(evols, erets, 'g', lw=4.0)
                # efficient frontier
    cx = np.linspace(0.0, 0.3)
    plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
                # capital market line
    plt.plot(opt[2], f(opt[2]), 'r*', markersize=15.0) 
    plt.grid(True)
    plt.axhline(0, color='k', ls='--', lw=2.0)
    plt.axvline(0, color='k', ls='--', lw=2.0)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    # plt.colorbar(label='Sharpe ratio')
    plt.savefig(PATH + 'port_opt_capital_market_line.png', dpi=300)
    plt.close()

    cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - f(opt[2])},
            {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    res = timeme(sco.minimize)(min_func_port, noa * [1. / noa,], method='SLSQP',
                           bounds=bnds, constraints=cons)
    print(res['x'].round(3))


if __name__ == '__main__':
    # norm_tests()
    # real_world_data()
    port_opt_theory()