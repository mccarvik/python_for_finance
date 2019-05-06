import sys, pdb
sys.path.append('/usr/share/doc')
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
mpl.rcParams['font.family'] = 'serif'
from pandas_datareader import data as web
from sklearn.decomposition import KernelPCA
import warnings
warnings.simplefilter('ignore')
import pymc3 as pm
import pytz
import datetime as dt
np.random.seed(1000)
from pymc3.distributions.timeseries import GaussianRandomWalk
import scipy.optimize as sco

PATH = '/home/ubuntu/workspace/python_for_finance/png/book_examples/ch11/'

def pca_data():
    symbols = ['IBM', 'AAPL', 'MSFT', 'XOM', 'MCD', 'KO', 'SPY']
    data = pd.DataFrame()
    for sym in symbols:
        data[sym] = web.DataReader(sym, data_source='google', start='1/1/2011')['Close']
    data = data.dropna()
    data[data.columns[:6]].head()
    return data

def applying_pca():
    data = pca_data()
    scale_function = lambda x: (x - x.mean()) / x.std()
    pca = KernelPCA().fit(data.apply(scale_function))
    print(len(pca.lambdas_))
    print(pca.lambdas_[:10].round())
    get_we = lambda x: x / x.sum()
    print(get_we(pca.lambdas_)[:10])
    print(get_we(pca.lambdas_)[:5].sum())

def constructing_pca_index():
    get_we = lambda x: x / x.sum()
    data = pca_data()
    spy = pd.DataFrame(data.pop('SPY'))
    scale_function = lambda x: (x - x.mean()) / x.std()
    pca = KernelPCA(n_components=1).fit(data.apply(scale_function))
    spy['PCA_1'] = pca.transform(-data)
    spy.apply(scale_function).plot(figsize=(7, 4))
    plt.savefig(PATH + 'pca1.png', dpi=300)
    plt.close()
    
    pca = KernelPCA(n_components=5).fit(data.apply(scale_function))
    pca_components = pca.transform(-data)
    weights = get_we(pca.lambdas_)
    spy['PCA_5'] = np.dot(pca_components, weights)
    spy.apply(scale_function).plot(figsize=(7, 4))
    plt.savefig(PATH + 'pca2.png', dpi=300)
    plt.close()

    pdb.set_trace()
    mpl_dates = mpl.dates.date2num(data.index.to_pydatetime())
    print(mpl_dates)
    plt.figure(figsize=(8, 4))
    plt.scatter(spy['PCA_5'], spy['SPY'], c=mpl_dates)
    lin_reg = np.polyval(np.polyfit(spy['PCA_5'],
                                    spy['SPY'], 1),
                                    spy['PCA_5'])
    plt.plot(spy['PCA_5'], lin_reg, 'r', lw=3)
    plt.grid(True)
    plt.xlabel('PCA_5')
    plt.ylabel('SPY')
    plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
                format=mpl.dates.DateFormatter('%d %b %y'))
    plt.savefig(PATH + 'pca3.png', dpi=300)
    plt.close()

    cut_date = '2011/7/1'
    early_pca = spy[spy.index < cut_date]['PCA_5']
    early_reg = np.polyval(np.polyfit(early_pca,
                    spy['SPY'][spy.index < cut_date], 1),
                    early_pca)
    late_pca = spy[spy.index >= cut_date]['PCA_5']
    late_reg = np.polyval(np.polyfit(late_pca,
                    spy['SPY'][spy.index >= cut_date], 1),
                    late_pca)
    plt.figure(figsize=(8, 4))
    plt.scatter(spy['PCA_5'], spy['SPY'], c=mpl_dates)
    plt.plot(early_pca, early_reg, 'r', lw=3)
    plt.plot(late_pca, late_reg, 'r', lw=3)
    plt.grid(True)
    plt.xlabel('PCA_5')
    plt.ylabel('SPY')
    plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
                    format=mpl.dates.DateFormatter('%d %b %y'))
    plt.savefig(PATH + 'pca4.png', dpi=300)
    plt.close()

def bayes_formula():
    x = np.linspace(0, 10, 500)
    y = 4 + 2 * x + np.random.standard_normal(len(x)) * 2
    reg = np.polyfit(x, y, 1)
    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, c=y, marker='v')
    plt.plot(x, reg[1] + reg[0] * x, lw=2.0)
    plt.colorbar()
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(PATH + 'bayes1.png', dpi=300)
    plt.close()
    print(reg)
    
    with pm.Model() as model: 
        # model specifications in PyMC3
        # are wrapped in a with-statement
        # define priors
        alpha = pm.Normal('alpha', mu=0, sd=20)
        beta = pm.Normal('beta', mu=0, sd=20)
        sigma = pm.Uniform('sigma', lower=0, upper=10)
        
        # define linear regression
        y_est = alpha + beta * x
        
        # define likelihood
        likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=y)
        
        # inference
        start = pm.find_MAP()
        # find starting value by optimization
        step = pm.NUTS(state=start)
        # instantiate MCMC sampling algorithm
        trace = pm.sample(100, step, start=start, progressbar=False)
        # draw 100 posterior samples using NUTS sampling
    
    print(trace[0])
    fig = pm.traceplot(trace, lines={'alpha': 4, 'beta': 2, 'sigma': 2})
    plt.savefig(PATH + 'bayes2.png', dpi=300)
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, c=y, marker='v')
    plt.colorbar()
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    for i in range(len(trace)):
        plt.plot(x, trace['alpha'][i] + trace['beta'][i] * x)
    plt.savefig(PATH + 'bayes3.png', dpi=300)
    plt.close()

def bayes_real_data():
    data = pd.DataFrame()
    symbols = ['GLD', 'GDX']
    for sym in symbols:
        data[sym] = web.DataReader(sym, data_source='google')['Close']
    print(data.info())
    data.plot(figsize=(7, 4))
    plt.savefig(PATH + 'bayes4.png', dpi=300)
    plt.close()
    
    print(data.ix[-1] / data.ix[0] - 1)
    print(data.corr())
    print(data.index)
    mpl_dates = mpl.dates.date2num(data.index.to_pydatetime())
    print(mpl_dates)
    plt.figure(figsize=(8, 4))
    plt.scatter(data['GDX'], data['GLD'], c=mpl_dates, marker='o')
    plt.grid(True)
    plt.xlabel('GDX')
    plt.ylabel('GLD')
    plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
                 format=mpl.dates.DateFormatter('%d %b %y'))
    plt.savefig(PATH + 'bayes5.png', dpi=300)
    plt.close()

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sd=20)
        beta = pm.Normal('beta', mu=0, sd=20)
        sigma = pm.Uniform('sigma', lower=0, upper=50)
        
        y_est = alpha + beta * data['GDX'].values
        
        likelihood = pm.Normal('GLD', mu=y_est, sd=sigma,
                               observed=data['GLD'].values)
        
        start = pm.find_MAP()
        step = pm.NUTS(state=start)
        trace = pm.sample(100, step, start=start, progressbar=False)

    pdb.set_trace()
    fig = pm.traceplot(trace)
    plt.savefig(PATH + 'bayes7.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(8, 4))
    plt.scatter(data['GDX'], data['GLD'], c=mpl_dates, marker='o')
    plt.grid(True)
    plt.xlabel('GDX')
    plt.ylabel('GLD')
    for i in range(len(trace)):
        plt.plot(data['GDX'], trace['alpha'][i] + trace['beta'][i] * data['GDX'])
    plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
                format=mpl.dates.DateFormatter('%d %b %y'))
    plt.savefig(PATH + 'bayes6.png', dpi=300)
    plt.close()

def bayes_randomwalk():
    # NOTE not compatible in python 3 version
    data = pd.DataFrame()
    symbols = ['GLD', 'GDX']
    for sym in symbols:
        data[sym] = web.DataReader(sym, data_source='google')['Close']
    
    pdb.set_trace()
    model_randomwalk = pm.Model()
    with model_randomwalk:
        # std of random walk best sampled in log space
        sigma_alpha, log_sigma_alpha = \
                model_randomwalk.TransformedVar('sigma_alpha', 
                                pm.Exponential.dist(1. / .02, testval=.1), 
                                pm.logtransform)
        sigma_beta, log_sigma_beta = \
                model_randomwalk.TransformedVar('sigma_beta', 
                                pm.Exponential.dist(1. / .02, testval=.1),
                                pm.logtransform)
    # to make the model more simple, we will apply the same coefficients
    # to 50 data points at a time
    
    subsample_alpha = 50
    subsample_beta = 50
    with model_randomwalk:
        alpha = GaussianRandomWalk('alpha', sigma_alpha**-2, 
                                   shape=len(data) / subsample_alpha)
        beta = GaussianRandomWalk('beta', sigma_beta**-2, 
                                  shape=len(data) / subsample_beta)
        
        # make coefficients have the same length as prices
        alpha_r = np.repeat(alpha, subsample_alpha)
        beta_r = np.repeat(beta, subsample_beta)
        print(len(data.dropna().GDX.values))  # a bit longer than 1,950
    
    with model_randomwalk:
        # define regression
        regression = alpha_r + beta_r * data.GDX.values[:1950]
    
        # assume prices are normally distributed,
        # the mean comes from the regression
        sd = pm.Uniform('sd', 0, 20)
        likelihood = pm.Normal('GLD', 
                               mu=regression, 
                               sd=sd, 
                               observed=data.GLD.values[:1950])

    with model_randomwalk:
        # first optimize random walk
        start = pm.find_MAP(vars=[alpha, beta], fmin=sco.fmin_l_bfgs_b)
        
        # sampling
        step = pm.NUTS(scaling=start)
        trace_rw = pm.sample(100, step, start=start, progressbar=False)
    print(np.shape(trace_rw['alpha']))

    part_dates = np.linspace(min(mpl_dates), max(mpl_dates), 39)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    plt.plot(part_dates, np.mean(trace_rw['alpha'], axis=0),
             'b', lw=2.5, label='alpha')
    for i in range(45, 55):
        plt.plot(part_dates, trace_rw['alpha'][i], 'b-.', lw=0.75)
    plt.xlabel('date')
    plt.ylabel('alpha')
    plt.axis('tight')
    plt.grid(True)
    plt.legend(loc=2)
    ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b %y') )
    ax2 = ax1.twinx()
    plt.plot(part_dates, np.mean(trace_rw['beta'], axis=0),
             'r', lw=2.5, label='beta')
    for i in range(45, 55):
        plt.plot(part_dates, trace_rw['beta'][i], 'r-.', lw=0.75)
    plt.ylabel('beta')
    plt.legend(loc=4)
    fig.autofmt_xdate()
    plt.savefig(PATH + 'bayes8.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.scatter(data['GDX'], data['GLD'], c=mpl_dates, marker='o')
    plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
                 format=mpl.dates.DateFormatter('%d %b %y'))
    plt.grid(True)
    plt.xlabel('GDX')
    plt.ylabel('GLD')
    x = np.linspace(min(data['GDX']), max(data['GDX'])) 
    for i in range(39):
        alpha_rw = np.mean(trace_rw['alpha'].T[i])
        beta_rw = np.mean(trace_rw['beta'].T[i]) 
        plt.plot(x, alpha_rw + beta_rw * x, color=plt.cm.jet(256 * i / 39))
    plt.savefig(PATH + 'bayes9.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    # applying_pca()
    # constructing_pca_index()
    # bayes_formula()
    # bayes_real_data()
    bayes_randomwalk()