"""
This util will grab data to and from the db with eod prices
"""
import pdb
import sys
import json
import datetime as dt
import pandas as pd
import requests
from pandas_datareader import data as pd_dr
from dateutil.relativedelta import relativedelta
# import pandas_datareader as dr
sys.path.append("/home/ec2-user/environment/python_for_finance/")
from utils.db_utils import DBHelper

FILE_PATH = '/home/ec2-user/environment/python_for_finance/data_grab/'
FILE_NAME = 'iex_available_stocks_2019_02_25.txt'

# px = HBIO
# morningstar = NHF
# is = TR

def quandl_load():
    """
    Load eod data from quandl
    """
    ticks = []
    start = dt.date(1999, 12, 1)
    end = dt.date(2019, 2, 26)
    with open(FILE_PATH + FILE_NAME, "r") as file:
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


def get_time_series(tick, start=None, end=dt.datetime.today()):
    """
    Pulls an individual time series from the API
    """
    if not start:
        start = dt.datetime.today() - relativedelta(years=5)
    data = pd_dr.DataReader(tick, 'iex', start, end)['close']
    return data


def get_list_of_symbols(typ=None):
    """
    Gets the list of potential stocks from IEX API
    """
    resp = requests.get('https://api.iextrading.com/1.0/ref-data/symbols').content.decode('utf-8')
    resp = json.loads(resp)
    if typ:
        resp = [ind_r for ind_r in resp if ind_r['type'] == typ]
    syms = [resp_sym['symbol'] for resp_sym in resp]

    with open(FILE_PATH + FILE_NAME, 'w') as file:
        for item in syms:
            file.write("%s\n" % item)


def load_db():
    """
    Gathers px data one by one through the ticks
    """
    ticks = []
    with open(FILE_PATH + FILE_NAME, "r") as file:
        for line in file:
            ticks.append(line.strip())

    count = 0
    for ind_t in ticks:
        # if ind_t < 'FGEN':
        #     count += 1
        #     print("skipping {}  already have data".format(ind_t))
        #     continue

        print("starting load for {}".format(ind_t))
        try:
            data = get_time_series(ind_t).reset_index()
            data['tick'] = ind_t
            data.columns = ['date', 'px', 'tick']
            send_to_db(data)
            print("Completed load for {}".format(ind_t))
            count += 1
            print("finished {}  of  {}  loads".format(count, len(ticks)))
        except Exception as exc:
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


        if where_clause:
            px_df = dbh.select('eod_px', where=where_clause).set_index(['date', 'tick'])
        else:
            px_df = dbh.select('eod_px').set_index(['date', 'tick'])

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
    # S_DT = dt.datetime(2013, 1, 1)
    # E_DT = dt.datetime(2019, 2, 22)
    # get_time_series('F')
    get_list_of_symbols('cs')
    # load_db()
    # quandl_load()
    # END_DT = dt.datetime.today
    # START_DT = END_DT - datetime.timedelta(days=365)
    # print(get_db_pxs(["A", "MSFT"]))
