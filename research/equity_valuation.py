import sys, pdb, time
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/home/ubuntu/workspace/ml_dev_work")

import numpy as np
import pandas as pd
import datetime as dt

from utils.db_utils import DBHelper

idx = ['date', 'ticker', 'month']

def income_state_model(ticks):
    data = get_ticker_info(['MCD'])
    data = prune_columns(data)
    data = clean_add_columns(data)
    pdb.set_trace()
    data_yoy = year_over_year(data)
    data = data.join(data_yoy.set_index(idx), how='inner')
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


def year_over_year(df):
    df_y = df.reset_index()
    years = list(df_y['date'])
    # df_y = df_y.set_index(['date', 'ticker', 'month'])
    df_yoy = pd.DataFrame(columns=idx + [c + '_yoy' for c in df.columns])
    for y in years:
        if y == min(years):
            continue
        year_df = df_y[df_y.date==y].drop(idx, axis=1).values
        last_y_df = df_y[df_y.date==str(int(y) - 1)].drop(idx, axis=1).values
        yoy = (year_df / last_y_df - 1) * 100
        yoy[abs(yoy) == np.inf] = 0
        where_are_NaNs = np.isnan(yoy)
        yoy[where_are_NaNs] = 0
        data = list(df_y[df_y.date == y].iloc[0][idx]) + list(yoy[0])
        df_yoy.loc[len(df_yoy)] = data
    return df_yoy


if __name__ == '__main__':
    income_state_model(['MCD'])