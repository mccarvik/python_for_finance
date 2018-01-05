import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb, requests, math, requests
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
from urllib.request import urlretrieve
from sklearn import linear_model


PATH = '/home/ubuntu/workspace/python_for_finance/png/book_examples/ch6/'

def intro():
    df = pd.DataFrame([10, 20, 30, 40], columns=['numbers'], index=['a', 'b', 'c', 'd'])
    # print(df.index)                     # index vlues
    # print(df.columns)                   # columns
    # print(df.ix['c'])                   # selection via index
    # print(df.ix[['a','d']])             # selection of multiple indices
    # print(df.ix[df.index[1:3]])         # selection vie index object
    # print(df.sum())                     # sum per column
    # print(df.apply(lambda x: x ** 2))   # square of every element
    # print (df ** 2)
    df['floats'] = (1.5, 2.5, 3.5, 4.5) 
    # print(df)
    # print(df['floats'])
    df['names'] = pd.DataFrame(['Yves', 'Guido', 'Felix', 'Francesc'], 
                                index=['d', 'a', 'b', 'c'])
    # print(df)
    # Notice side effect that index gets replaced by numbered index
    # print(df.append({'numbers': 100, 'floats': 5.75, 'names': 'Henry'}, ignore_index=True))
    df = df.append(pd.DataFrame({'numbers': 100, 'floats': 5.75, 'names': 'Henry'}, index=['z',]))
    # print(df)
    
    # Join method --> does SQL type, set theory combinations (right, left, inner, outer, etc.)
    print(df.join(pd.DataFrame([1, 4, 9, 16, 25], 
                        index=['a', 'b', 'c', 'd', 'y'],
                        columns=['squares',])))
    df =  df.join(pd.DataFrame([1, 4, 9, 16, 25], 
                        index=['a', 'b', 'c', 'd', 'y'],
                        columns=['squares',]), how='outer')
    print(df)
    print(df[['numbers', 'squares']].mean())
    print(df[['numbers', 'squares']].std())

def second_steps():
    a = np.random.standard_normal((9,4))
    a.round(6)
    df = pd.DataFrame(a)
    df.columns = [['No1', 'No2', 'No3', 'No4']]
    # print(df['No2'][3])
    dates = pd.date_range('2015-1-1', periods=9, freq='M')
    # print(dates)
    df.index = dates
    
    # can go np to pandas and pandas to np
    print(np.array(df).round(6))
    # basic analytics
    print(df.sum())
    print(df.mean())
    print(df.cumsum())
    print(df.describe())
    print(np.sqrt(df))
    # Can sum even though there are NaNs
    print(np.sqrt(df).sum())
    # Can also use matplotlib (assuming you have the right interpreter)
    # df.cumsum().plot(lw=2.0)
    
    print(type(df['No1']))
    # df['No1'].cumsum().plot(style='r', lw=2)
    
    # Group By Operations
    df['Quarter'] = ['Q1,', 'Q1,', 'Q1,', 'Q2', 'Q2', 'Q2', 'Q3', 'Q3', 'Q3',]
    groups = df.groupby('Quarter')
    print(groups.mean())
    print(groups.max())
    print(groups.size())
    df['Odd_Even'] = ['Odd', 'Even', 'Odd', 'Even', 'Odd', 'Even', 'Odd', 'Even', 'Odd']
    groups = df.groupby(['Quarter', 'Odd_Even'])
    print(groups.size())
    print(groups.mean())
    
def fin_data():
    SPY = web.DataReader(name='SPY', data_source='google', start='2000-1-1')
    print(SPY.info())
    print(SPY.tail())
    plt.figure(figsize=(9, 4))
    plt.plot(SPY['Close'])
    plt.savefig(PATH + 'spy.png', dpi=300)
    plt.close()
    
    SPY['Ret_Loop'] = 0.0
    # for i in range(1, len(SPY)):
    #     SPY['Ret_Loop'][i] = np.log(SPY['Close'][i] / SPY['Close'][i-1])
    print(SPY[['Close', 'Ret_Loop']].tail())
    SPY['Return'] = np.log(SPY['Close'] / SPY['Close'].shift(1))
    print(SPY[['Close', 'Return']].tail())
    # del SPY['Ret_Loop']
    SPY['Mov_Vol'] = pd.rolling_std(SPY['Return'], window=252) * math.sqrt(252)
    
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(SPY['Close'])
    axarr[1].plot(SPY['Return'])
    axarr[2].plot(SPY['Mov_Vol'])
    axarr[0].legend(loc=0)
    axarr[1].legend(loc=0)
    axarr[2].legend(loc=0)
    plt.savefig(PATH + 'close_return.png', dpi=300)
    plt.close()
    
    SPY['42d'] = pd.rolling_mean(SPY['Close'], window=42)
    SPY['252d'] = pd.rolling_mean(SPY['Close'], window=252)
    print(SPY[['Close', '42d', '252d']].tail())
    
    plt.figure(figsize=(9, 4))
    plt.plot(SPY[['Close', '42d', '252d']])
    plt.savefig(PATH + 'mvg_avg.png', dpi=300)
    plt.close()
    
def regr_analysis():
    es_url = 'https://www.stoxx.com/download/historical_values/hbrbcpe.txt'
    vs_url = 'https://www.stoxx.com/download/historical_values/h_vstoxx.txt'
    urlretrieve(es_url, '/home/ubuntu/workspace/python_for_finance/data/es.txt')
    urlretrieve(vs_url, '/home/ubuntu/workspace/python_for_finance/data/vs.txt')
    lines = open('/home/ubuntu/workspace/python_for_finance/data/es.txt', 'r').readlines()
    lines = [line.replace(' ', '') for line in lines]
    new_file = open('/home/ubuntu/workspace/python_for_finance/data/es50.txt', 'w')
    new_file.writelines('date' + lines[3][:-1] + ';DEL' + lines[3][-1])
    new_file.writelines(lines[4:])
    new_file.close()
    new_lines = open('/home/ubuntu/workspace/python_for_finance/data/es50.txt', 'r').readlines()
    es = pd.read_csv('/home/ubuntu/workspace/python_for_finance/data/es50.txt', index_col=0, 
                    parse_dates=True, sep=';', dayfirst=True)
    print(np.round(es.tail()))
    del es['DEL']
    print(es.info())
    cols = ['SX5P', 'SX5E', 'SXXP', 'SXXE', 'SXXF', 'SXXA', 'DK5F', 'DKXF']
    es = pd.read_csv(es_url, index_col=0, parse_dates=True, sep=';', dayfirst=True,
                    header=None, skiprows=4, names=cols)
    print(es.tail())
    vs = pd.read_csv('./data/vs.txt', index_col=0, header=2, parse_dates=True, 
                    sep=',', dayfirst=True)
    print(vs.info())
    data = pd.DataFrame({'EUROSTOXX' : es['SX5E'][es.index > dt.datetime(1999, 1, 1)]})
    data = data.join(pd.DataFrame({'VSTOXX' : vs['V2TX'][vs.index > dt.datetime(1999, 1, 1)]}))
    data = data.fillna(method='ffill')
    print(data.info())
    print(data.tail())
    
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(data['EUROSTOXX'])
    axarr[1].plot(data['VSTOXX'])
    axarr[0].legend(loc=0)
    axarr[1].legend(loc=0)
    plt.savefig(PATH + 'eurostoxx_vstoxx.png', dpi=300)
    plt.close()
    
    rets = np.log(data / data.shift(1))
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(rets['EUROSTOXX'])
    axarr[1].plot(rets['VSTOXX'])
    axarr[0].legend(loc=0)
    axarr[1].legend(loc=0)
    plt.savefig(PATH + 'eurostoxx_vstoxx_rets.png', dpi=300)
    plt.close()
    
    rets = rets.dropna()
    rets_corr = rets
    rets = rets[:100]
    xdat = rets['EUROSTOXX']
    ydat = rets['VSTOXX']
    
    pdb.set_trace()
    regr = linear_model.LinearRegression()
    regr.fit(np.transpose(np.matrix(xdat)), np.transpose(np.matrix(ydat)))
    
    # BRUTAL issue with sm.OLS import
    # model = sm.OLS(y=ydat, x=xdat)
    # print(model)
    # print(model.beta)
    
    plt.plot(xdat, ydat, 'r.')
    ax = plt.axis()
    x = np.linspace(ax[0], ax[1] + 0.01)
    plt.plot(x, regr.intercept_[0] + regr.coef_[0][0] * x, 'b', lw=2)
    plt.grid(True)
    plt.axis('tight')
    plt.xlabel('EURO STOXX 50 returns')
    plt.ylabel('VSTOXX returns')
    plt.savefig(PATH + 'regr.png', dpi=300)
    plt.close()

    pdb.set_trace()
    print(rets.corr())
    plt.plot(pd.rolling_corr(rets_corr['EUROSTOXX'], rets_corr['VSTOXX'], window=252))
    plt.savefig(PATH + 'corr.png', dpi=300)
    plt.close()
    
def hf():
    # url1 = 'http://hopet.netfonds.no/quotes/posdump.php?'
    # url2 = 'date=%s%s%s&paper=AAPL.0&csv_format=csv'
    # url = url1 + url2
    # year = dt.date.today().year
    # month = dt.date.today().month
    # day = dt.date.today().day
    # days = [day, day-1, day-2]
    url = 'https://www.google.com/finance/getprices?i=300&p=3d&f=d,o,h,l,c,v&res=cpct&q=IBM'
    csv = requests.get(url).content.decode('utf-8').splitlines()[8:]
    csv = [c.split(',') for c in csv]
    pdb.set_trace()
    ibm = pd.DataFrame(csv)
    ibm.columns = ['DATE', 'CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME']
    print(ibm.info())
    
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(ibm['CLOSE'])
    axarr[1].plot(ibm['VOLUME'])
    axarr[0].legend(loc=0)
    axarr[1].legend(loc=0)
    plt.savefig(PATH + 'intradey.png', dpi=300)
    plt.close()
    
    # Need the netfonds data for this
    # Resampling smooths out the choppy point areas
    # ibm_resam = ibm.resample(rule='5min', how='mean')
    # print(np.round(ibm_resam.head(), 2))
    
    # use apply function to apply a func to the whole data set
    plt.figure(figsize=(9, 4))
    plt.plot(ibm['CLOSE'].fillna(method='ffill').apply(reversal))
    plt.savefig(PATH + 'reversal.png', dpi=300)
    plt.close()
    
def reversal(x):
    return 2 * 95 - float(x)

if __name__ == "__main__":
    # intro()
    # second_steps()
    # fin_data()
    regr_analysis()
    # hf()