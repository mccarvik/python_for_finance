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
        data = makeAPICall(ticks[0], 'is')
        data_cum = dataCumColumns(data)
        data_chg = period_chg(data)
        data_margin = margin_df(data)[['grossProfit', 'cogs', 'rd', 'sga', 'mna', 'other_exp']]
        data_est = modelEst(data_cum, data_chg, data_margin, ['2018', ticks[0], '03'])
        pdb.set_trace()
        # reorganizing columns
        data = reorgCols(data_cum, data_chg)
    else:
        data = get_ticker_info(['MCD'])
        data = prune_columns(data)
        data = clean_add_columns(data)
        data_chg = period_chg(data)
    # data = data.join(data_chg.set_index(idx), how='inner')
    pdb.set_trace()
    print()


def modelEst(cum, chg, margin, dt_idx):
    df_est = pd.DataFrame()
    # some cleanup
    margin = margin.reset_index()[margin.reset_index().date != 'TTM'].set_index(idx)
    cum = cum.reset_index()[cum.reset_index().month != ''].set_index(idx)
    
    # next qurater est is equal to revenue * average est margin over the last year
    while True:
        n_idx = list(cum.iloc[-1].name)
        n_idx[2] = "0" + str(int(n_idx[2]) + 3) + 'E'
        n_data = n_idx + list(margin[-5:-1].mean())
        t_df = pd.DataFrame(dict((key, value) for (key, value) in zip(idx+list(margin.columns), n_data)), columns=idx+list(margin.columns), index=[0]).set_index(idx)
        margin = margin.append(t_df)
        n_cum_dict = {k: v for k, v in zip(idx, n_idx)}
        n_cum_dict['revenue'] = cum[-5:-1]['revenue'].mean()
        for c in ['cogs', 'rd', 'sga', 'grossProfit', 'mna', 'other_exp']:
            n_cum_dict[c] = margin.loc[tuple(n_idx)][c] * n_cum_dict['revenue']
        pdb.set_trace()
        n_cum_dict['totalOperatingCost'] = n_cum_dict['rd'] + n_cum_dict['sga'] + n_cum_dict['cogs'] + n_cum_dict['mna'] + n_cum_dict['other_exp']
        n_cum_dict['EBIT'] = n_cum_dict['revenue'] - n_cum_dict['totalOperatingCost']    # operating income
        t_df = pd.DataFrame(n_cum_dict, index=[0]).set_index(idx)
        cum = cum.append(t_df)


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
    df_chg = pd.DataFrame(columns=idx + list(df.columns))
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
    
    # need this to add year over year for single year model
    yoy = (df_y.drop(idx, axis=1).loc[len(df_y)-1].values / df_y.drop(idx, axis=1).loc[0].values - 1) * 100
    yoy[abs(yoy) == np.inf] = 0
    where_are_NaNs = np.isnan(yoy)
    yoy[where_are_NaNs] = 0
    data = ['YoY', df.reset_index().ticker[0], ''] + list(yoy)
    df_chg.loc[len(df_chg)] = data
    df_chg = df_chg.set_index(idx)
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
        columns = IS_COLS
        headers = is_replacements(headers)
        data = [[columns[h]] + d[1:] for h, d in zip(headers, data)]
    
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
    # Need this to fill in columns not included
    for i in columns.values():
        if i not in data.columns:
            data[i] = 0
    data = data.set_index(idx)
    return data    


def dataCumColumns(data):
    tick = data.reset_index()['ticker'][0]
    data = data.drop([('TTM', tick, '')])
    H1 = data.iloc[-4] + data.iloc[-3]
    M9 = H1 + data.iloc[-2]
    Y1 = M9 + data.iloc[-1]
    data = data.reset_index()
    data.loc[len(data)] = ['H1', tick, ''] + list(H1.values)
    data.loc[len(data)] = ['M9', tick, ''] + list(M9.values)
    data.loc[len(data)] = ['Y1', tick, ''] + list(Y1.values)
    data = data.reindex([0,1,2,5,3,6,4,7])
    data = data.set_index(idx)
    return data


def reorgCols(cums, chgs):
    # just orders the columsn for use, VERY rigid
    return cums.append(chgs).reset_index().reindex([0, 1, 8, 2, 9, 3, 4, 10, 5, 6, 11, 7, 12]).set_index(idx)
    

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
    # income_state_model(['MSFT'])
    # income_state_model(['MCD'])
    income_state_model(['CSCO'])