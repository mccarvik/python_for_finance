import sys
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas_datareader as dr
from utils.db_utils import DBHelper, restart


def get_time_series(tick, start=None, end=dt.datetime.today()):
    if not start:
        start = dt.datetime.today() - relativedelta(years=5)
    data = dr.DataReader(tick, 'iex', start, end)['close']
    return data


def load_db():
    ticks = []
    with open("/home/ubuntu/workspace//ml_dev_work/utils/snp_ticks_2018_02_13.txt", "r") as f:
        for line in f:
            ticks.append(line.strip())
    
    for ind_t in ticks:
        data = get_time_series(ind_t).reset_index()
        data['tick'] = ind_t
        data.columns = ['date', 'px', 'tick']
        sendToDB(data)


def sendToDB(df):
    with DBHelper() as db:
        db.connect()
        table = 'eod_px'
        import pdb; pdb.set_trace()
        prim_keys = ['tick', 'date']
        for ind, vals in df.iterrows():
            val_dict = vals.to_dict()
            db.upsert(table, val_dict, prim_keys)

if __name__ == '__main__':
    # S_DT = dt.datetime(2013, 1, 1)
    E_DT = dt.datetime(2019, 2, 22)
    get_time_series('F')
    load_db()
