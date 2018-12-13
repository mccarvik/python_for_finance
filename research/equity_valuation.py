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
from dx.frame import get_year_deltas

idx = ['date', 'ticker', 'month']
sum_cols = ['revenue', 'intExp', 'shares', 'EBT', 'netIncome', 'taxes', 'EPS', 'cogs', 'sga', 'rd']

hist_quarterly_map = {
    'operatingIncome' : 'EBIT',
    'totalLiabilities' : 'totalLiab',
    'cashAndShortTermInv' : 'totalCash',
    'netInterestOtherMargin' : 'otherInc'
}

def income_state_model(ticks, mode='db'):
    if mode == 'api':
        data = makeAPICall(ticks[0], 'is')
        # reorganizing columns
        # data = reorgCols(data_cum, data_chg)
    else:
        data = get_ticker_info(ticks, 'morningstar_monthly_is')
        data_bs = get_ticker_info(ticks, 'morningstar_monthly_bs')[['totalCash', 'totalLiab']]
        data_hist = get_ticker_info(ticks, 'morningstar')
        # join on whatever data you need
        data = data.join(data_bs)

    data = removeEmptyCols(data)
    data_cum = dataCumColumns(data)
    data_chg = period_chg(data)
    data_margin = margin_df(data)[['grossProfit', 'cogs', 'rd', 'sga', 'mna', 'otherExp']]
    # data_est = modelEst(data_cum, data_hist, data_chg, data_margin)
    data_est = modelEstOLS(data_cum, data_hist, data_chg, data_margin)
    
    
    # data = prune_columns(data)
    # data = clean_add_columns(data)
    # data_chg = period_chg(data)
    # data = data.join(data_chg.set_index(idx), how='inner')
    pdb.set_trace()
    print()


def modelEstOLS(cum, hist, chg, margin):
    df_est = pd.DataFrame()
    # some cleanup
    margin = margin.reset_index()[margin.reset_index().date != 'TTM'].set_index(idx)
    cum = cum.reset_index()[cum.reset_index().month != ''].set_index(idx)
    
    # next qurater est is equal to revenue * average est margin over the last year
    for i in range(4):
        n_idx = list(cum.iloc[-1].name)
        n_idx = getNextQuarter(n_idx)
        
        n_cum_dict = {k: v for k, v in zip(idx, n_idx)}
        n_hist_dict = {k: v for k, v in zip(idx, n_idx)}
        # Assume these are static
        total_debt = cum.iloc[-1]['totalLiab']
        cash_and_inv = cum.iloc[-1]['totalCash']
        
        #########
        # Use OLS to get projected values
        #########
        
        # need to convert from margin to gross
        pdb.set_trace()
        hist_cols_conv = ['cogs', 'rd', 'sga', 'other', 'netInterestOtherMargin', 'shares']
        for cc in ['totalLiabilities', 'cashAndShortTermInv'] + hist_cols_conv:
            hist[cc] = hist['totalAssets'] * hist[cc]
        
        for cc in ['revenue', 'operatingIncome'] + hist_cols_conv:
            xs = hist.reset_index()[['date','month']]
            slope, yint = ols_calc(xs, hist[cc])
            start = dt.datetime(int(xs.values[0][0]), int(xs.values[0][1]), 1).date()
            new_x = get_year_deltas([start, dt.datetime(int(n_idx[0]), int(n_idx[2][:-1]), 1).date()])[-1]
            # divide by four for quarterly
            cc = hist_quarterly_map.get(cc, cc)
            # n_cum_dict[cc] = (yint + new_x * slope) / 4
            n_hist_dict[cc] = (yint + new_x * slope)
        
        # 0.6 = about corp rate , 0.02 = about yield on cash, 0.25 = 1/4 of the year
        # n_cum_dict['intExp'] = total_debt * 0.25 * 0.7 - cash_and_inv * 0.25 * 0.02
        # n_cum_dict['EBT'] = n_cum_dict['EBIT'] - n_cum_dict['intExp'] + n_cum_dict['otherInc']
        
        n_hist_dict['intExp'] = total_debt * 0.25 * 0.7 - cash_and_inv * 0.25 * 0.02
        n_hist_dict['EBT'] = n_hist_dict['EBIT'] - n_hist_dict['intExp'] + n_hist_dict['otherInc']
        
        # average tax rate of the last 5 years
        # n_cum_dict['taxes'] = (cum[-5:-1]['taxes'] / cum[-5:-1]['EBT']).mean() * n_cum_dict['EBT']
        # n_cum_dict['netIncome'] = n_cum_dict['EBT'] - n_cum_dict['taxes']
        
        n_hist_dict['taxes'] = (cum[-5:-1]['taxes'] / cum[-5:-1]['EBT']).mean() * n_hist_dict['EBT']
        n_hist_dict['netIncome'] = n_hist_dict['EBT'] - n_hist_dict['taxes']
        
        # Assume change over the last year continues for next quarter - may need to adjust this per company
        # n_cum_dict['shares'] = ((((cum.iloc[-1]['shares'] / cum.iloc[-5]['shares']) - 1) / 4) + 1) * cum.iloc[-1]['shares']
        # n_cum_dict['EPS'] = n_cum_dict['netIncome'] / n_cum_dict['shares']
        
        n_hist_dict['shares'] = ((((cum.iloc[-1]['shares'] / cum.iloc[-5]['shares']) - 1) / 4) + 1) * cum.iloc[-1]['shares']
        n_hist_dict['EPS'] = n_hist_dict['netIncome'] / n_hist_dict['shares']
        
        # t_df = pd.DataFrame(n_cum_dict, index=[0]).set_index(idx)
        t_df = pd.DataFrame(n_hist_dict, index=[0]).set_index(idx)
        cum = cum.append(t_df)
        
    pdb.set_trace()
    cum[sum_cols]
    return cum
        


def modelEst(cum, hist, chg, margin):
    df_est = pd.DataFrame()
    # some cleanup
    margin = margin.reset_index()[margin.reset_index().date != 'TTM'].set_index(idx)
    cum = cum.reset_index()[cum.reset_index().month != ''].set_index(idx)
    
    # next qurater est is equal to revenue * average est margin over the last year
    for i in range(4):
        n_idx = list(cum.iloc[-1].name)
        n_idx = getNextQuarter(n_idx)
        n_data = n_idx + list(margin[-5:-1].mean())
        t_df = pd.DataFrame(dict((key, value) for (key, value) in zip(idx+list(margin.columns), n_data)), columns=idx+list(margin.columns), index=[0]).set_index(idx)
        margin = margin.append(t_df)
        
        n_cum_dict = {k: v for k, v in zip(idx, n_idx)}
        n_cum_dict['revenue'] = cum[-5:-1]['revenue'].mean()
        
        #########
        # Use mean of previous few years to get there
        #########
        for c in ['cogs', 'rd', 'sga', 'grossProfit', 'mna', 'otherExp']:
            n_cum_dict[c] = margin.loc[tuple(n_idx)][c] * n_cum_dict['revenue']
        n_cum_dict['operatingCost'] = n_cum_dict['rd'] + n_cum_dict['sga'] + n_cum_dict['mna'] + n_cum_dict['otherExp']
        n_cum_dict['EBIT'] = n_cum_dict['revenue'] - n_cum_dict['operatingCost'] - n_cum_dict['cogs']   # operating income
        # Need to update these when we do balance sheet
        total_debt = cum.iloc[-1]['totalLiab']
        cash_and_inv = cum.iloc[-1]['totalCash']
        # 0.6 = about corp rate , 0.02 = about yield on cash, 0.25 = 1/4 of the year 
        n_cum_dict['intExp'] = total_debt * 0.25 * 0.7 - cash_and_inv * 0.25 * 0.02
        # just assume average of last year, gonna be very specific company to company
        n_cum_dict['otherInc'] = cum[-5:-1]['otherInc'].mean()
        n_cum_dict['EBT'] = n_cum_dict['EBIT'] - n_cum_dict['intExp'] + n_cum_dict['otherInc']
        # average tax rate of the last year
        n_cum_dict['taxes'] = (cum[-5:-1]['taxes'] / cum[-5:-1]['EBT']).mean() * n_cum_dict['EBT']
        n_cum_dict['netIncome'] = n_cum_dict['EBT'] - n_cum_dict['taxes']
        
        # Assume change over the last year continues for next quarter - may need to adjust this per company
        n_cum_dict['shares'] = ((((cum.iloc[-1]['shares'] / cum.iloc[-5]['shares']) - 1) / 4) + 1) * cum.iloc[-1]['shares']
        n_cum_dict['sharesBasic'] = ((((cum.iloc[-1]['sharesBasic'] / cum.iloc[-5]['sharesBasic']) - 1) / 4) + 1) * cum.iloc[-1]['sharesBasic']
        
        n_cum_dict['EPS'] = n_cum_dict['netIncome'] / n_cum_dict['shares']
        n_cum_dict['EPSBasic'] = n_cum_dict['netIncome'] / n_cum_dict['sharesBasic']
        
        # Assume cash and liabilities are static
        n_cum_dict['totalLiab'] = total_debt
        n_cum_dict['totalCash'] = cash_and_inv
        
        t_df = pd.DataFrame(n_cum_dict, index=[0]).set_index(idx)
        cum = cum.append(t_df)
        
    # clean up df
    cum = cum.fillna(0)
    empty_cols = [c for c in list(cum.columns) if all(v == 0 for v in cum[c])]
    cum = cum[list(set(cum.columns) - set(empty_cols))]
    pdb.set_trace()
    return cum


def getNextQuarter(index):
    tick = index[1]
    y = int(index[0])
    m = int(index[2].replace("E","")) + 3
    if m > 12:
        m -= 12
        y += 1
    if m < 10:
        m = "0" + str(m)
    return [str(y), tick, str(m)+"E"]


def clean_add_columns(df):
    # Edit the margin based columns
    for c in ['cogs', 'sga', 'rd']:
        df[c] = (df[c] / 100) * df['revenue']
    df['operatingCost'] = df['sga'] + df['rd']
    df['Taxes'] = df['EBT'] - df['netIncome']
    return df
    

def get_ticker_info(ticks, table, dates=None):
    # Temp to make testing quicker
    t0 = time.time()
    # tickers = pd.read_csv('/home/ubuntu/workspace/ml_dev_work/utils/dow_ticks.csv', header=None)
    with DBHelper() as db:
        db.connect()
        # df = db.select('morningstar', where = 'date in ("2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016")')
        lis = ''
        for t in ticks:
            lis += "'" + t + "', "
        df = db.select(table, where = 'ticker in (' + lis[:-2] + ')')
        
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


def dataCumColumns(data):
    tick = data.reset_index()['ticker'][0]
    try:
        data = data.drop([('TTM', tick, '')])
    except:
        # no TTM included
        pass
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
    

def ols_calc(xs, ys, n_idx=None):
    xs = [dt.datetime(int(x[0]), int(x[1]), 1).date() for x in xs.values]
    start = xs[0]
    xs = get_year_deltas(xs)
    A = np.vstack([xs, np.ones(len(xs))]).T
    slope, yint = np.linalg.lstsq(A, ys.values)[0]
    # new_x = get_year_deltas([start, dt.datetime(int(n_idx[0]), int(n_idx[2][:-1]), 1).date()])[-1]
    return (slope, yint)


def removeEmptyCols(df):
    return df.dropna(axis='columns', how='all')


if __name__ == '__main__':
    income_state_model(['MSFT'])
    # income_state_model(['MCD'])
    # income_state_model(['CSCO'])
    # income_state_model(['FLWS'], mode='db')
    # income_state_model(['FLWS'], mode='api')