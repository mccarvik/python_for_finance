import sys
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import pdb, time, requests, csv
import numpy as np
import pandas as pd
import datetime as dt

from res_utils import *
from utils.db_utils import DBHelper

idx = ['date', 'ticker', 'month']

def income_state_model(ticks, mode='api'):
    if mode == 'api':
        data = makeAPICall('MCD', 'is')
    else:
        data = get_ticker_info(['MCD'])
        data = prune_columns(data)
        data = clean_add_columns(data)
    data_chg = period_chg(data)
    pdb.set_trace()
    data_margin = margin_df(data)
    # data = data.join(data_chg.set_index(idx), how='inner')
    pdb.set_trace()
    print()


def clean_add_columns(df):
    # Edit the margin based columns
    for c in ['cogs', 'sga', 'rd']:
        df[c] = (df[c] / 100) * df['revenue']
    df['operatingCost'] = df['sga'] + df['rd']
    df['Taxes'] = df['EBT'] - df['netIncome']
    return df
    

def get_ticker_info(ticks, dates=None):
    # Temp to make testing quicker
    t0 = time.time()
    # tickers = pd.read_csv('/home/ubuntu/workspace/ml_dev_work/utils/dow_ticks.csv', header=None)
    with DBHelper() as db:
        db.connect()
        # df = db.select('morningstar', where = 'date in ("2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016")')
        lis = ''
        for t in ticks:
            lis += "'" + t + "', "
        df = db.select('morningstar', where = 'ticker in (' + lis[:-2] + ')')
        
    # Getting Dataframe
    t1 = time.time()
    print("Done Retrieving data, took {0} seconds".format(t1-t0))
    return df.set_index(['date', 'ticker', 'month'])


def prune_columns(df):
    desired_cols = ['revenue', 'cogs', 'grossProfit', 'sga', 'rd', 'operatingIncome', 'netInterestOtherMargin', 'other',
                    'EBTMargin', 'taxRate', 'taxesPayable', 'netIncome', 'shares', 'dividendPerShare', 'trailingEPS', 'EBT']
    return df[desired_cols]


def period_chg(df):
    df_y = df.reset_index()
    df_y = df_y[df_y.date != 'TTM']
    years = list(df_y['date'] + df_y['month'])
    # df_y = df_y.set_index(['date', 'ticker', 'month'])
    df_chg = pd.DataFrame(columns=idx + [c + '_chg' for c in df.columns])
    for y in years:
        if y == min(years):
            last_y = y
            continue
        year_df = df_y[(df_y.date==y[:4]) & (df_y.month==y[4:])].drop(idx, axis=1).values
        last_y_df = df_y[(df_y.date==last_y[:4]) & (df_y.month==last_y[4:])].drop(idx, axis=1).values
        yoy = (year_df / last_y_df - 1) * 100
        yoy[abs(yoy) == np.inf] = 0
        where_are_NaNs = np.isnan(yoy)
        yoy[where_are_NaNs] = 0
        data = list(df_y[(df_y.date==y[:4]) & (df_y.month==y[4:])].iloc[0][idx]) + list(yoy[0])
        df_chg.loc[len(df_chg)] = data
        last_y = y
    return df_chg


def margin_df(df):
    data = df.apply(lambda x: x / x['revenue'], axis=1)
    data['taxes'] = df['taxes'] / df['EBT']     # tax rate presented as percentage of pre tax income
    return data


def makeAPICall(ticker, sheet='bs', per=3, col=10, num=3):
    # Use this for quarterly info
    # Period can be 3 or 12 for quarterly vs annual
    # Sheet can be bs = balance sheet, is = income statement, cf = cash flow statement
    # Column year can be 5 or 10, doesnt really work
    # 1 = None 2 = Thousands 3 = Millions 4 = Billions
    url = 'http://financials.morningstar.com/ajax/ReportProcess4CSV.html?t={0}&reportType={1}&period={2}&dataType=A&order=asc&columnYear={3}&number={4}'.format(ticker, sheet, per, col, num)
    urlData = requests.get(url).content.decode('utf-8')
    if 'Error reading' in urlData:
        # try one more time
        time.sleep(5)
        urlData = requests.get(url).content.decode('utf-8')
    if 'Error reading' in urlData:
        print('API issue')
        return None
        
    cr = csv.reader(urlData.splitlines(), delimiter=',')
    data = []
    for row in list(cr):
        data.append(row)
    # Remove empty rows
    data = [x for x in data if x != []]
    data = [x for x in data if len(x) != 1]
    # Remove dates
    dates = data[0][1:]
    data = data[1:]
    # Remove certain strings from headers (USD, Mil, etc) and removing first 2 lines
    headers = [d[0] for d in data]
    # Replace headers with better col names
    if sheet == 'is':
        headers = is_replacements(headers)
    data = [[IS_COLS[h]] + d[1:] for h, d in zip(headers, data)]
    data = pd.DataFrame(data).transpose()
    cols = data.iloc[0]
    data = data[1:]
    data = data.apply(pd.to_numeric)
    data.columns = cols
    years = [d.split('-')[0] for d in dates]
    months = [d.split('-')[1] for d in dates[:-1]+["-"]]
    data['date'] = years
    data['month'] = months
    data['ticker'] = ticker
    data = data.set_index(idx)
    return data    


def is_replacements(headers):
    first = False
    new_headers = []
    for h in headers:
        if not first:
            if h == 'Basic':
                new_headers.append('EPS' + h)
            else:
                if h == 'Diluted':
                    new_headers.append('EPS' + h)
                    first = True
                else:
                    new_headers.append(h)
        else:
            if h == 'Basic':
                new_headers.append('Shares' + h)
            else:
                if h == 'Diluted':
                    new_headers.append('Shares' + h)
                else:
                    new_headers.append(h)
    return new_headers


if __name__ == '__main__':
    income_state_model(['MCD'])