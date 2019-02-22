"""
This util will grab data to and from the db with eod prices
"""
import sys
import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas_datareader as dr
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
from utils.db_utils import DBHelper


def get_time_series(tick, start=None, end=dt.datetime.today()):
    """
    Pulls an individual time series from the API
    """
    if not start:
        start = dt.datetime.today() - relativedelta(years=5)
    data = dr.DataReader(tick, 'iex', start, end)['close']
    return data


def load_db():
    """
    Gathers px data one by one through the ticks
    """
    ticks = []
    with open("/home/ubuntu/workspace//ml_dev_work/utils/snp_ticks_2018_02_13.txt", "r") as file:
        for line in file:
            ticks.append(line.strip())

    for ind_t in ticks:
        data = get_time_series(ind_t).reset_index()
        data['tick'] = ind_t
        data.columns = ['date', 'px', 'tick']
        send_to_db(data)


def send_to_db(data):
    """
    Sends the px data to the DB
    """
    with DBHelper() as dbh:
        dbh.connect()
        table = 'eod_px'
        prim_keys = ['tick', 'date']
        for ind, vals in data.iterrows():
            val_dict = vals.to_dict()
            dbh.upsert(table, val_dict, prim_keys)

if __name__ == '__main__':
    # S_DT = dt.datetime(2013, 1, 1)
    # E_DT = dt.datetime(2019, 2, 22)
    # get_time_series('F')
    load_db()
