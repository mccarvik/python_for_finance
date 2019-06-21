"""
Calculate the size factor and ranking amongst peers for equity valuation
"""
import sys
import pdb
import time
import datetime as dt
sys.path.append("/home/ec2-user/environment/python_for_finance/")
from data_grab.eod_px_util import FILE_PATH, FILE_NAME_STOCK
from utils.db_utils import DBHelper

def calc_small_v_big(trade_dt):
    """
    calculates the 12-2 return, aka return from 12 months ago to one month ago
    """
    # Get parameters for query
    stocks = []
    read_date = "20190619"
    with open(FILE_PATH + FILE_NAME_STOCK.format(read_date), "r") as file:
        for line in file:
            stocks.append(line.strip())

    # stocks = ['A', 'AA', 'AAPL']
    # where_clause = ("date >= '{}' and date <= '{}' and tick in"
    #                 "('A', 'AAPL', 'AA')".format(start_dt, end_dt))
    where_clause = "(year = '2018' or year = '2019') and tick in ('"
    tick_clause = "', '".join(stocks) + "')"
    where_clause += tick_clause

    pdb.set_trace()
    print("starting query")
    time0 = time.time()
    with DBHelper() as dbh:
        dbh.connect()
        tick_df = dbh.select('fin_ratios',
                             where=where_clause, 
                             cols=['tick', 'year', 'month', 'market_cap'])
        tick_df = tick_df.set_index(['tick', 'year', 'month'])
    time1 = time.time()
    print("Done Retrieving data, took {0} seconds".format(time1-time0))

    tick_df = tick_df['market_cap'].sort_values(ascending=False).to_frame()
    tick_df['percentile'] = (tick_df['market_cap'].rank() / len(tick_df))
    tick_df.to_csv("static/output/small_v_big_{}.csv"
                   "".format(trade_dt.strftime("%Y%m%d")))
    return tick_df


if __name__ == '__main__':
    END_DT = dt.datetime.today()
    RET = calc_small_v_big(END_DT)
    print(RET)
