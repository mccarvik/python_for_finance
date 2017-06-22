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


PATH = '/home/ubuntu/workspace/python_for_finance/png/ch11/'

def pca_data():
    symbols = ['ADS.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BEI.DE',
           'BMW.DE', 'CBK.DE', 'CON.DE', 'DAI.DE', 'DB1.DE',
           'DBK.DE', 'DPW.DE', 'DTE.DE', 'EOAN.DE', 'FME.DE',
           'FRE.DE', 'HEI.DE', 'HEN3.DE', 'IFX.DE', 'LHA.DE',
           'LIN.DE', 'LXS.DE', 'MRK.DE', 'MUV2.DE', 'RWE.DE',
           'SAP.DE', 'SDF.DE', 'SIE.DE', 'TKA.DE', 'VOW3.DE',
           '^GDAXI']
    data = pd.DataFrame()
    for sym in symbols:
        data[sym] = web.DataReader(sym, data_source='yahoo')['Close']
    data = data.dropna()
    dax = pd.DataFrame(data.pop('^GDAXI'))
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
    data = pca_data()
    dax = pd.DataFrame(data.pop('^GDAXI'))
    scale_function = lambda x: (x - x.mean()) / x.std()
    pca = KernelPCA(n_components=1).fit(data.apply(scale_function))
    dax['PCA_1'] = pca.transform(-data)
    dax.apply(scale_function).plot(figsize=(7, 4))
    plt.savefig(PATH + 'pca1.png', dpi=300)
    plt.close()
    
    pca = KernelPCA(n_components=5).fit(data.apply(scale_function))
    pca_components = pca.transform(-data)
    weights = get_we(pca.lambdas_)
    dax['PCA_5'] = np.dot(pca_components, weights)
    dax.apply(scale_function).plot(figsize=(7, 4))
    plt.savefig(PATH + 'pca2.png', dpi=300)
    plt.close()

    mpl_dates = mpl.dates.date2num(data.index.to_pydatetime())
    print(mpl_dates)
    plt.figure(figsize=(8, 4))
    plt.scatter(dax['PCA_5'], dax['^GDAXI'], c=mpl_dates)
    lin_reg = np.polyval(np.polyfit(dax['PCA_5'],
                                    dax['^GDAXI'], 1),
                                    dax['PCA_5'])
    plt.plot(dax['PCA_5'], lin_reg, 'r', lw=3)
    plt.grid(True)
    plt.xlabel('PCA_5')
    plt.ylabel('^GDAXI')
    plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
                format=mpl.dates.DateFormatter('%d %b %y'))
    plt.savefig(PATH + 'pca3.png', dpi=300)
    plt.close()

    cut_date = '2011/7/1'
    early_pca = dax[dax.index < cut_date]['PCA_5']
    early_reg = np.polyval(np.polyfit(early_pca,
                    dax['^GDAXI'][dax.index < cut_date], 1),
                    early_pca)
    late_pca = dax[dax.index >= cut_date]['PCA_5']
    late_reg = np.polyval(np.polyfit(late_pca,
                    dax['^GDAXI'][dax.index >= cut_date], 1),
                    late_pca)
    plt.figure(figsize=(8, 4))
    plt.scatter(dax['PCA_5'], dax['^GDAXI'], c=mpl_dates)
    plt.plot(early_pca, early_reg, 'r', lw=3)
    plt.plot(late_pca, late_reg, 'r', lw=3)
    plt.grid(True)
    plt.xlabel('PCA_5')
    plt.ylabel('^GDAXI')
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
    plt.figure(figsize=(8, 8))
    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, c=y, marker='v')
    plt.colorbar()
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    for i in range(len(trace)):
        plt.plot(x, trace['alpha'][i] + trace['beta'][i] * x)
    plt.savefig(PATH + 'bayes2.png', dpi=300)
    plt.close()

def bayes_real_data():
    data = pd.DataFrame()
    symbols = ['GLD', 'GDX']
    for sym in symbols:
        data[sym] = web.DataReader(sym, data_source='yahoo')['Adj Close']
    print(data.info())
    data.plot(figsize=(7, 4))
    plt.savefig(PATH + 'bayes3.png', dpi=300)
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
    plt.savefig(PATH + 'bayes3.png', dpi=300)
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

    fig = pm.traceplot(trace)
    plt.figure(figsize=(8, 8))
    plt.savefig(PATH + 'bayes4.png', dpi=300)
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
    plt.savefig(PATH + 'bayes5.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    pca_data()