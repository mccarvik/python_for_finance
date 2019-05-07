import pdb
import sys
import time
import requests
import io
import json
import pandas as pd
import numpy as np
import datetime as dt
from threading import Thread
sys.path.append("/home/ec2-user/environment/python_for_finance/")
from data_grab import ms_dg_helper
from utils.db_utils import DBHelper, restart

success = []
failure = []

FILE_PATH = '/home/ec2-user/environment/python_for_finance/data_grab/'
FILE_NAME = 'fmp_available_stocks_{}.txt'

# quora: https://www.quora.com/Is-there-a-free-API-for-financial-statement-information-for-all-public-companies
# fmp inf0 --> https://financialmodelingprep.com/developer/docs#Financial-Ratios
fmp_financial_ratios_url = "https://financialmodelingprep.com/api/financial-ratios/{}?period=quarter&datatype=csv"

def get_available_ticks():
    url = "https://financialmodelingprep.com/api/stock/list/all?datatype=json"
    raw = requests.get(url).content
    data = json.loads(raw)
    data = pd.DataFrame(data)['Ticker'].values
    pdb.set_trace()
    tod = dt.datetime.today().strftime("%Y%m%d")
    with open(FILE_PATH + FILE_NAME.format(tod), 'w') as file:
        for item in data:
            file.write("%s\n" % item)


def get_fin_ratios(tickers=None):
    tasks = []
    if not tickers:
        with open("/home/ec2-user/environment/python_for_finance/data_grab/fmp_available_stocks_20190507.txt", "r") as f:
            for line in f:
                tasks.append(line.strip())
    else:
        tasks = tickers
    t0 = time.time()
    threads = []
    print("total stocks: {}".format(len(tasks)))
    
    try:
        ct = 0
        start = False
        starter = 'SPY'
        for t in tasks:
            if not start and t != starter:
                continue
            else:
                start = True
            
            if ct % 25 == 0:
                print(str(ct) + " stocks completed so far")
            try:
                fin_ratios_api(t)
                success.append(t)
            except:
                # pdb.set_trace()
                failure.append(t)
                print("Failed " + t + "\t")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print("Error in task loop: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
            ct+=1
    except Exception as e:
        # pdb.set_trace()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Error in getData: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
    
    t1 = time.time()
    text_file = open("Failures.txt", "w")
    text_file.write(("\t").join(failure))
    print("\t".join(success))
    print("Done Retrieving data, took {0} seconds".format(t1-t0))


def fin_ratios_api(tick):
    url = fmp_financial_ratios_url.format(tick)
    raw = requests.get(url).content
    data = pd.read_csv(io.StringIO(raw.decode('utf-8')))
    try:
        data = data.drop('TTM', axis=1)
    except Exception as exc:
        print(exc)
        print("May be an etf where this isnt applicable")
    data = data.set_index('Ratios').transpose().reset_index()
    data['month'] = data['index'].str.slice(5)
    data['year'] = data['index'].str.slice(0,4)
    data['tick'] = tick
    data = data.drop('index', axis=1)
    data = data.set_index(['tick', 'year', 'month'])
    sendToDB(data, 'fin_ratios', ['tick', 'year', 'month'])
    

def getData(tickers=None):
    # Getting all the tickers from text file
    tasks = []
    if not tickers:
        with open("/home/ec2-user/environment/python_for_finance/snp_ticks_2018_02_13.txt", "r") as f:
            for line in f:
                tasks.append(line.strip())
    else:
        tasks = tickers
    t0 = time.time()
    threads = []
    tasks = [t for t in tasks if t not in ms_dg_helper.remove_ticks_ms]
    tasks = [t for t in tasks if t not in ms_dg_helper.remove_ticks_dr]
    
    try:
        # for running multithreaded: starts the thread and 'joins it' so we will wait for all to finish
        # API ACTING WEIRD WITH MULTITHREADING
        # ct = 0
        # chunks = 30
        # while ct < len(tasks):
        #     if ct+chunks > len(tasks):
        #         add = len(tasks - chunks)
        #     sub_tasks = tasks[ct:ct+chunks]
        #     for t in sub_tasks:
        #         threads.append(Thread(target=makeAPICall, args=(t,)))
        #     [t.start() for t in threads]
        #     [t.join() for t in threads]
        #     ct += chunks
        #     threads = []
        #     print(str(ct) + " stocks completed so far")
            
        
        # for running single threaded
        # tasks = tasks[2600:]
        ct = 0
        for t in tasks:
            if ct % 25 == 0:
                print(str(ct) + " stocks completed so far")
            try:
                makeAPICall(t)
                success.append(t)
            except:
                # pdb.set_trace()
                failure.append(t)
                print("Failed " + t + "\t")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print("Error in task loop: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
            ct+=1
    except Exception as e:
        # pdb.set_trace()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Error in getData: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
    
    t1 = time.time()
    text_file = open("Failures.txt", "w")
    text_file.write(("\t").join(failure))
    print("\t".join(success))
    print("Done Retrieving data, took {0} seconds".format(t1-t0))


def makeAPICall(tick):
    pdb.set_trace()
    print("Trying " + tick + "\t")
    try:
        url = 'http://financials.morningstar.com/ajax/exportKR2CSV.html?t=' + tick
        urlData = requests.get(url).content.decode('utf-8')
        if urlData == '':
            # try one more time
            time.sleep(5)
            urlData = requests.get(url).content.decode('utf-8')
        cr = csv.reader(urlData.splitlines(), delimiter=',')
        data = []
        for row in list(cr):
            data.append(row)
        # Remove dates
        dates = data[2][1:]
        # Remove empty rows
        data = [x for x in data if x != []]
        # Remove certain strings from headers (USD, Mil, etc)
        headers = [d[0] for d in data]
        for repl in ms_dg_helper.remove_strs:
            headers = [h.replace(repl,"").strip() for h in headers]
        data = [[h] + d[1:] for h, d in zip(headers, data)]
        data = pd.DataFrame(data)
        pruneData(data, dates, tick)
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        # Chance Company Not published on Morningstar
        print("Error in API Call, probably no Morningstar data, for {3}  {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj, tick))
        raise
        
    print("Succeeded " + tick + "\t")
    success.append(tick)

    
def pruneData(df, dates, tick):
    df = df.set_index(0)
    df = df.transpose()
    # if tick == "AHS":
    #     print("")
    # rename columns and throw away random extra columns
    df = df.rename(columns=ms_dg_helper.COL_MAP)
    try:
        col_list = [c for c in ms_dg_helper.COL_MAP.values() if c in df.columns]
        df = df[col_list]
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Error with column mapping for {3}: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj, tick))
    
    # The TTM data may be off as that last column has different time windows
    years = [d.split("-")[0] for d in dates[:-1]] + [str(datetime.date.today().year)]
    months = [d.split("-")[1] for d in dates[:-1]] + [str(datetime.date.today().month)]
    df['date'] = years
    df['month'] = months
    df['ticker'] = tick
    df = df.set_index(['ticker', 'date'])
    df = cleanData(df)
    try:
        df = addCustomColumns(df)
        sendToDB(df)
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Did not load data for {3}: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj, tick))
        raise


def cleanData(df):
    # Need this for commas
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',',''), errors='ignore'))
    df = df.fillna(0)
    # Need this as the API returns two rows for "Revenue", one we dont need
    revenue = df['revenue'].ix[:,0]
    df = df.drop('revenue', 1)
    df['revenue'] = pd.Series(revenue)
    return df


def addCustomColumns(df, market_upd=False):
    if df.reset_index()['ticker'][0] == 'ATVI':
        # pdb.set_trace()
        print()
    start = datetime.date(int(df.index.get_level_values('date')[0])-10, int(df['month'].values[0]), 1)
    end_date_ls = [int(d) for d in datetime.date.today().strftime('%Y-%m-%d').split("-")]
    end = datetime.date(end_date_ls[0], end_date_ls[1], end_date_ls[2])
    try:
        url = "https://www.quandl.com/api/v1/datasets/WIKI/{0}.csv?column=4&sort_order=asc&trim_start={1}&trim_end={2}".format(df.index.get_level_values('ticker')[0], start, end)
        qr = pd.read_csv(url)
        qr['Date'] = qr['Date'].astype('datetime64[ns]')
        # quotes = DataReader(df.index.get_level_values('ticker')[0], 'yahoo', start, end, pause=1)['Close']
        # quotes = DataReader(df.index.get_level_values('ticker')[0], 'yahoo', start, end, pause=1)['Close']
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Could not read time series data for {3}: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj, df.index.get_level_values('ticker')[0]))
        raise
        
    try:
        df = addBasicCustomCols(df, qr)
        df = addGrowthCustomCols(df, qr)
        df = addTimelineCustomCols(df, qr)
    except Exception as e:
        # pdb.set_trace()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Issue adding columns {3}: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj, df.index.get_level_values('ticker')[0]))
        raise
    
    if market_upd:
        market = DataReader(".INX", 'google', start, end, pause=1)['Close']
        market = DataReader('^GSPC','yahoo', start, end,pause=1)['Close']
        market.to_csv('/home/ubuntu/workspace/finance/app/static/docs/market.csv')
        quotes = pd.DataFrame(quotes)
        market.columns = ['Date', 'market']
        market.set_index('Date')
        quotes['market'] = market
    else:
        market = pd.read_csv('/home/ubuntu/workspace/finance/app/static/docs/market.csv')
        market.columns = ['Date', 'market']
        market['Date'] = market['Date'].apply(pd.to_datetime)
        market = market.set_index('Date')
        qr = pd.merge(qr.set_index('Date'), market, left_index=True, right_index=True)
    df = calcBetas(df, qr)
    '''
    Still need to do:
    'enterpriseToEbitda'
    'ebitdaMargins'
    'ebitda',
    'shortRatio'
    '''
    return df


def addTimelineCustomCols(df, qr, vol_window=252):
    quotes = qr.reset_index()
    
    quotes['mv_avg_50'] = quotes['Close'].rolling(center=False,window=50).mean()
    quotes['mv_avg_200'] = quotes['Close'].rolling(center=False,window=200).mean()
    quotes['mv_avg_for_vol'] = quotes['Close'].rolling(center=False,window=vol_window).mean()  # Need this for volatility, avg price over the period
    quotes['vol_stdev'] = quotes['Close'].rolling(window=vol_window, center=False).std()

    df['last_day'] = ''
    df['50DayMvgAvg'] = ''
    df['200DayMvgAvg'] = ''
    df['mv_avg_for_vol'] = ''
    df['volatility'] = ''
    df = df.reset_index().set_index(['date', 'month'])
    for ix, row in df.iterrows():
        try:
            df.at[ix, 'last_day'] =  quotes[quotes['Date'] >= datetime.date(int(ix[0]),int(ix[1]),1)].iloc[0]['Date']
            df.at[ix, '50DayMvgAvg'] =  float(quotes[quotes['Date'] == df.ix[ix].last_day]['mv_avg_50'])
            df.at[ix, '200DayMvgAvg'] =  float(quotes[quotes['Date'] == df.ix[ix].last_day]['mv_avg_200'])
            df.at[ix, 'mv_avg_for_vol'] =  float(quotes[quotes['Date'] == df.ix[ix].last_day]['mv_avg_for_vol'])
            df.at[ix, 'volatility'] = float(quotes[quotes['Date'] == df.ix[ix].last_day]['vol_stdev']) / float(quotes[quotes['Date'] == df.ix[ix].last_day]['mv_avg_for_vol']) * 100
        except Exception as e:
            df.at[ix, 'last_day'] =  np.nan
            df.at[ix, '50DayMvgAvg'] =  np.nan
            df.at[ix, '200DayMvgAvg'] =  np.nan
            df.at[ix, 'mv_avg_for_vol'] =  np.nan
            df.at[ix, 'volatility'] = np.nan
            print("May not have the quote data necessary for these calcs")
    
    df['sharpeRatio'] = df['3yrReturn'] / df['volatility']
    df['downsideVol'] = downside_vol(df, qr, vol_window) / df['mv_avg_for_vol'] * 100 
    df['sortinoRatio'] = df['3yrReturn'] / df['downsideVol']
    df = df.drop(['mv_avg_for_vol', 'last_day'], axis=1)
    return df


def calcBetas(df, quotes, years_hist=3):
    # TODO: Need to pick a smaller volatility range to fine tune the beta here
    qs = quotes.dropna().reset_index()
    qs = qs.drop('change', axis=1)
    betas = []; corrs = []
    for ind, vals in df.iterrows():
        try:
            # default 3 year history
            np_array = qs[(qs['Date'] >= datetime.date(int(ind[1])-years_hist,int(vals['month']),1)) & (qs['Date'] <= datetime.date(int(ind[1]),int(vals['month']),1))].values
            market = np_array[:,1]                      # market returns are column zero from numpy array
            stock = np_array[:,2]                       # stock returns are column one from numpy array
            corr = scipy.stats.pearsonr(stock, market)[0]
            covariance = np.cov(stock,market)           # Calculate covariance between stock and market
            beta = covariance[0,1]/covariance[1,1]
            corrs.append(corr)
            betas.append(beta)
        except:
            corrs.append(0)
            betas.append(0)
    df['beta'] = betas
    df['marketCorr'] = corrs
    df['treynorRatio'] = df['5yrReturn'] / df['beta']
    return df


def downside_vol(df, qr, vol_window):
    qr['change'] = qr['Close'].pct_change(1)
    df['downsideVol'] = ''
    for ix, row in df.iterrows():
        try:
            date_ix = qr[qr['Date'] == df.ix[ix].last_day].index.values[0]
            start_ix = date_ix - vol_window if date_ix - vol_window > 0 else 0
            q_wind = qr[start_ix : date_ix]
            df.at[ix, 'downsideVol'] = q_wind[q_wind['change'] < 0]['Close'].std()
        except Exception as e:
            df.at[ix, 'downsideVol'] = np.nan
            print("May not have the quote data necessary for these calcs")
    return df['downsideVol']


def addBasicCustomCols(df, qr):
    # Need this for times where no prices available
    curPrices = []
    for ind, x in df.iterrows():
        y = qr[qr['Date'] <= datetime.date(int(x.name[1]),int(x['month']),1)]
        if not y.empty:
            curPrices.append(y.iloc[-1]['Close'])
        else:
            curPrices.append(0)
    df['currentPrice'] = curPrices
    # Remove all periods with 0 price, aka no price data for that area (company probabyl didnt exist)
    df = df[df['currentPrice'] != 0]
    df['revenuePerShare'] = df['revenue'] / df['shares'] 
    df['divYield'] = df['dividendPerShare'] / df['currentPrice'] * 100
    df['trailingPE'] = df['currentPrice'] / df['trailingEPS']
    df['priceToBook'] = df['currentPrice'] / df['bookValuePerShare']
    df['priceToSales'] = df['currentPrice'] / df['revenuePerShare']
    df['priceToCashFlow'] = df['currentPrice'] / df['freeCashFlowPerShare']
    df['grossProfit'] = (df['grossMargin']/100) * df['revenue']
    df['marketCapital'] = df['shares'] * df['currentPrice']
    df['totalAssets'] = (df.bookValuePerShare * df.shares) / (df.totalEquity / 100)
    df['enterpriseValue'] = df['marketCapital'] + (df['totalLiabilities'] * df['totalAssets']) - (df['cashAndShortTermInv'] * df['totalAssets'])
    df['enterpriseToRevenue'] = df['enterpriseValue'] / df['revenue']
    df['EBT'] = (df['operatingIncome'] - (df['netInterestOtherMargin']/100 * df['revenue']))    # aka Pretax Income
    return df

    
def addGrowthCustomCols(df, qr):
    rev_growth = []; eps_growth = []
    for ind, vals in df.iterrows():
        try:
            last = df.loc[(df.index.get_level_values('date') == str(int(ind[1])-1))]
            if last.empty:
                rev_growth.append(0)
                eps_growth.append(0)
            else:
                rev_growth.append(float((vals['revenue'] / last['revenue'] - 1) * 100))
                eps_growth.append(float((vals['trailingEPS'] / last['trailingEPS'] - 1) * 100))
        except:
            rev_growth.append(0)
            eps_growth.append(0)
    
    df['revenueGrowth'] = rev_growth
    df['epsGrowth'] = eps_growth
    df['pegRatio'] = df['trailingPE'] / df['epsGrowth']
    
    yr1_ret = []; yr3_ret = []; yr5_ret = []; yr10_ret = []; ytd_ret =[]; min52 = []; max52 = []
    for ind, vals in df.iterrows():
        try:
            yr1q = qr[qr['Date'] >= datetime.date(int(ind[1])-1,int(vals['month']),1)].iloc[0]['Close']
            yr1_ret.append(((vals['currentPrice'] / yr1q) - 1) * 100)
        except:
            yr1_ret.append(0)
        try:
            yr3q = qr[qr['Date'] >= datetime.date(int(ind[1])-3,int(vals['month']),1)].iloc[0]['Close']
            yr3_ret.append(((vals['currentPrice'] / yr3q)**(1/3) - 1) * 100)
        except:
            yr3_ret.append(0)
        try:
            yr5q = qr[qr['Date'] >= datetime.date(int(ind[1])-5,int(vals['month']),1)].iloc[0]['Close']
            yr5_ret.append(((vals['currentPrice'] / yr5q)**(1/5) - 1) * 100)
        except:
            yr5_ret.append(0)
        try:
            yr10q = qr[qr['Date'] >= datetime.date(int(ind[1])-10,int(vals['month']),1)].iloc[0]['Close']
            yr10_ret.append(((vals['currentPrice'] / yr10q)**(1/10) - 1) * 100)
        except:
            yr10_ret.append(0)
        try:
            yrytd = qr[qr['Date'] >= datetime.date(int(ind[1])-10,1,1)].iloc[0]['Close']
            ytd_ret.append(((vals['currentPrice'] / yrytd - 1) * 100))
        except:
            ytd_ret.append(0)
        try:
            min52.append(min(qr[(qr['Date'] >= datetime.date(int(ind[1])-1,int(vals['month']),1)) & (qr['Date'] <= datetime.date(int(ind[1]),int(vals['month']),1))]['Close']))
            max52.append(max(qr[(qr['Date'] >= datetime.date(int(ind[1])-1,int(vals['month']),1)) & (qr['Date'] <= datetime.date(int(ind[1]),int(vals['month']),1))]['Close']))
        except:
            min52.append(0)
            max52.append(0)

    df['ytdReturn'] = ytd_ret
    df['1yrReturn'] = yr1_ret
    df['3yrReturn'] = yr3_ret
    df['5yrReturn'] = yr5_ret
    df['10yrReturn'] = yr10_ret
    df['52WeekLow'] = min52
    df['52WeekHigh'] = max52
    return df


def sendToDB(df, table, prim_keys):
    with DBHelper() as db:
        db.connect()
        for ind, vals in df.reset_index().iterrows():
            val_dict = vals.to_dict()
            db.upsert(table, val_dict, prim_keys)


if __name__ == "__main__":
    # getData(['MCD'])
    # get_fin_ratios(['MSFT'])
    get_fin_ratios()
    # get_available_ticks()