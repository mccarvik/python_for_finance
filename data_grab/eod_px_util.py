"""
This util will grab data to and from the db with eod prices
"""
# import pdb
import sys
import time
import json
import datetime as dt
import pandas as pd
import requests
# import pandas_datareader as dr
sys.path.append("/home/ec2-user/environment/python_for_finance/")
from utils.db_utils import DBHelper

FILE_PATH = '/home/ec2-user/environment/python_for_finance/data_grab/'
FILE_NAME_SYM = 'fmp_avail_symbols_{}.txt'
FILE_NAME_STOCK = 'fmp_avail_stocks_{}.txt'


def quandl_load():
    """
    Load eod data from quandl
    """
    ticks = []
    start = dt.date(1999, 12, 1)
    end = dt.date(2019, 2, 26)
    with open(FILE_PATH + FILE_NAME_SYM, "r") as file:
        for line in file:
            ticks.append(line.strip())

    count = 0
    for ind_t in ticks:
        if ind_t < 'ACC':
            count += 1
            print("skipping {}  already have data".format(ind_t))
            continue

        print("starting load for {}".format(ind_t))
        try:
            url = "https://www.quandl.com/api/v1/datasets/WIKI/{0}.csv?column=4&sort_order=asc&trim_start={1}&trim_end={2}".format(ind_t, start, end)
            data = pd.read_csv(url)
            data['tick'] = ind_t
            data.columns = ['date', 'px', 'tick']
            send_to_db(data)
            print("Completed load for {}".format(ind_t))
            count += 1
            print("finished {}  of  {}  loads".format(count, len(ticks)))
        except Exception as exc:
            count += 1
            print("FALIED for {}".format(ind_t))
            print(exc)


def get_time_series(tick):
    """
    Pulls an individual time series from the API
    """
    url = "https://financialmodelingprep.com/api/v3/historical-price-full/{}?serietype=line".format(tick)
    raw = requests.get(url).content
    data = json.loads(raw)['historical']
    data = pd.DataFrame(data)
    return data


def get_list_of_stocks():
    """
    Gets the list of potential stocks from IEX API
    """
    today = dt.datetime.today().strftime("%Y%m%d")
    with DBHelper() as dbh:
        dbh.connect()
        tick_df = dbh.select('fin_ratios', cols=['DISTINCT(tick)'])

    syms = tick_df['tick'].values
    with open(FILE_PATH + FILE_NAME_STOCK.format(today), 'w') as file:
        for item in syms:
            file.write("%s\n" % item)


def get_list_of_symbols():
    """
    Gets the list of potential stocks from IEX API
    """
    today = dt.datetime.today().strftime("%Y%m%d")
    resp = requests.get('https://financialmodelingprep.com/api/v3/company/stock/list').content.decode('utf-8')
    resp = json.loads(resp)['symbolsList']
    syms = [resp_sym['symbol'] for resp_sym in resp]

    with open(FILE_PATH + FILE_NAME_SYM.format(today), 'w') as file:
        for item in syms:
            file.write("%s\n" % item)


def load_db(start=None):
    """
    Gathers px data one by one through the ticks
    """
    ticks = []
    read_date = "20190619"
    with open(FILE_PATH + FILE_NAME_SYM.format(read_date), "r") as file:
        for line in file:
            ticks.append(line.strip())

    data = pd.DataFrame()
    count = 0
    batch = 50
    already_have = True
    for ind_t in ticks:
        if ind_t == 'NBB':
            already_have = False
        if already_have:
            count += 1
            print("skipping {}  already have data".format(ind_t))
            continue

        print("starting load for {}".format(ind_t))
        try:
            t_data = get_time_series(ind_t, start).reset_index()
            t_data['tick'] = ind_t
            t_data = t_data.drop(['index'], axis=1)
            t_data.columns = ['px', 'date', 'tick']
            if start:
                t_data = t_data[t_data.date > start.strftime("%Y-%m-%d")]
            # send_to_db(t_data)
            if not t_data.empty:
                if data.empty:
                    data = t_data
                else:
                    data = pd.concat([data, t_data])
            else:
                print("Failed for {}".format(ind_t))

            if count % batch == 0 and not data.empty:
                send_to_db(data)
                loaded_ticks = list(data['tick'].unique())
                print("Completed load for {}".format(", ".join(loaded_ticks)))
                print("Finished {}  of  {}  loads".format(count, len(ticks)))
                data = pd.DataFrame()
            count += 1
            print("Gathered data for {}".format(ind_t))
        except KeyError as exc:
            print("FALIED for {}".format(ind_t))
            print(exc)


def send_to_db(data):
    """
    Sends the px data to the DB
    """
    with DBHelper() as dbh:
        dbh.connect()
        table = 'eod_px'
        prim_keys = ['tick', 'date']
        for _, vals in data.iterrows():
            val_dict = vals.to_dict()
            dbh.upsert(table, val_dict, prim_keys)


def get_db_pxs(ticks=None, s_date=None, e_date=None):
    """
    Will grab data from the db for the given parameters
    """
    with DBHelper() as dbh:
        dbh.connect()
        where_clause = ''
        if ticks:
            lis = ''
            for ind_t in list(ticks):
                lis += "'" + ind_t + "', "
            where_clause = 'tick in (' + lis[:-2] + ')'

        if s_date:
            if where_clause:
                where_clause += ' and date >= "{}"'.format(s_date.strftime('%Y-%m-%d'))
            else:
                where_clause = 'date >= "{}"'.format(s_date.strftime('%Y-%m-%d'))

        if e_date:
            if where_clause:
                where_clause += ' and date <= "{}"'.format(e_date.strftime('%Y-%m-%d'))
            else:
                where_clause = 'date <= "{}"'.format(e_date.strftime('%Y-%m-%d'))

        time0 = time.time()
        print("Starting retrieving data")
        if where_clause:
            px_df = dbh.select('eod_px', where=where_clause).set_index(['date', 'tick'])
        else:
            px_df = dbh.select('eod_px').set_index(['date', 'tick'])
        time1 = time.time()
        print("Done Retrieving data, took {0} seconds".format(time1-time0))
        # final_df = pd.DataFrame()
        # for ind_t in set(list(px_df['tick'].values)):
        #     temp_df = px_df[px_df.tick == ind_t][['px']]
        #     temp_df.columns = [ind_t]
        #     if final_df.empty:
        #         final_df = temp_df
        #     else:
        #         final_df = final_df.merge(temp_df, left_index=True, right_index=True)
        final_df = px_df
        return final_df


if __name__ == '__main__':
    S_DT = dt.datetime(2019, 5, 1)
    # E_DT = dt.datetime(2019, 2, 22)

    # get_time_series('F')
    # get_list_of_symbols()
    # get_list_of_stocks()

    load_db(start=S_DT)
    # END_DT = dt.datetime.today
    # START_DT = END_DT - datetime.timedelta(days=365)
    # print(get_db_pxs(["A", "MSFT"]))
