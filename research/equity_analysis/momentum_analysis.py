"""
Calculate the momentum return and ranking amongst peers for
equity valuation
"""
import sys
# import pdb
import time
import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta
sys.path.append("/home/ec2-user/environment/python_for_finance/")
from data_grab.eod_px_util import FILE_PATH, FILE_NAME_STOCK
from utils.db_utils import DBHelper

def calc_mom_return(trade_dt):
    """
    calculates the 12-2 return, aka return from 12 months ago to one month ago
    """
    # Get parameters for query
    start_dt = (trade_dt - relativedelta(days=365)).strftime("%Y-%m-%d")
    end_dt = (trade_dt - relativedelta(days=30)).strftime("%Y-%m-%d")
    stocks = []
    read_date = "20190619"
    with open(FILE_PATH + FILE_NAME_STOCK.format(read_date), "r") as file:
        for line in file:
            stocks.append(line.strip())

    # stocks = ['A', 'AA', 'AAPL']
    # where_clause = ("date >= '{}' and date <= '{}' and tick in"
    #                 "('A', 'AAPL', 'AA')".format(start_dt, end_dt))
    where_clause = ("date >= '{}' and date <= '{}' and tick in ('"
                    "".format(start_dt, end_dt))
    tick_clause = "', '".join(stocks) + "')"
    where_clause += tick_clause

    print("starting query")
    time0 = time.time()
    with DBHelper() as dbh:
        dbh.connect()
        tick_df = dbh.select('eod_px',
                             where=where_clause).set_index(['tick', 'date'])
    time1 = time.time()
    print("Done Retrieving data, took {0} seconds".format(time1-time0))

    ind_stock_df = pd.DataFrame()
    for ind_stock in stocks:
        try:
            if ind_stock_df.empty:
                ind_stock_df = tick_df.loc[ind_stock].rename(columns={'px':ind_stock})
            else:
                ind_stock_df[ind_stock] = tick_df.loc[ind_stock]
        except KeyError:
            print("no price data for {}".format(ind_stock))

    time2 = time.time()
    print("Done combining dataframes, took {0} seconds".format(time2-time1))
    annual_return = (ind_stock_df.iloc[-1] / ind_stock_df.iloc[0]
                     - 1).sort_values(ascending=False).to_frame().dropna()
    annual_return.index.names = ['tick']
    annual_return.columns = ['return']
    annual_return['percentile'] = (annual_return['return'].rank() 
                                  / len(annual_return))
    annual_return.to_csv("static/output/momentum_returns_{}.csv"
                         "".format(trade_dt.strftime("%Y%m%d")))
    return annual_return


if __name__ == '__main__':
    END_DT = dt.datetime.today()
    RET = calc_mom_return(END_DT)
    print(RET)
