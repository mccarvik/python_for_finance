"""
Script to download varous datasets from finanical modeling prep website
"""
import pdb
import sys
import time
import io
import json
import warnings
import datetime as dt
import requests
import pandas as pd
sys.path.append("/home/ec2-user/environment/python_for_finance/")
from utils.db_utils import DBHelper, get_ticker_table_data
from data_grab.fmp_helper import map_columns, add_px_ret_to_fr
from data_grab.eod_px_util import get_db_pxs

warnings.filterwarnings("ignore")
SUCCESS = []
FAILURE = []

FILE_PATH = '/home/ec2-user/environment/python_for_finance/data_grab/'
FILE_NAME = 'fmp_available_stocks_{}.txt'

# quora: https://www.quora.com/Is-there-a-free-API-for-financial-statement-information-for-all-public-companies
# fmp inf0 --> https://financialmodelingprep.com/developer/docs#Financial-Ratios


def get_available_ticks():
    """
    Gets all available tickers from FMP and places them in a file
    """
    url = "https://financialmodelingprep.com/api/stock/list/all?datatype=json"
    raw = requests.get(url).content
    data = json.loads(raw)
    data = pd.DataFrame(data)['Ticker'].values
    tod = dt.datetime.today().strftime("%Y%m%d")
    with open(FILE_PATH + FILE_NAME.format(tod), 'w') as file:
        for item in data:
            file.write("%s\n" % item)


def get_fmp_data(dataset="fin_ratios", tickers=None):
    """
    Retrieves data from FMP API
    """
    tasks = []
    if not tickers:
        with open("/home/ec2-user/environment/python_for_finance/data_grab/fmp_available_stocks_20190507.txt", "r") as file:
            for line in file:
                tasks.append(line.strip())
    else:
        tasks = tickers
    time0 = time.time()
    print("total stocks: {}".format(len(tasks)))

    count = 0
    for tick in tasks:
        print(tick)
        if count % 25 == 0:
            print(str(count) + " stocks completed so far")
        try:
            DATA_MAP[dataset][0](tick)
            SUCCESS.append(tick)
        except KeyError:
            FAILURE.append(tick)
            print("Failed " + tick + "\t")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("Error in task loop: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
        except Exception as exc:
            pdb.set_trace()
            print(exc)
        count += 1

    time1 = time.time()
    text_file = open("Failures.txt", "w")
    text_file.write(("\t").join(FAILURE))
    print("\t".join(SUCCESS))
    print("Done Retrieving data, took {0} seconds".format(time1-time0))


def fin_ratios_api(tick):
    """
    reach out to the fin_ratios API on FMP
    """
    # get fin ratios
    url = DATA_MAP['fin_ratios'][1][0].format(tick)
    raw = requests.get(url).content
    data = pd.read_csv(io.StringIO(raw.decode('utf-8')))
    try:
        data = data.drop('TTM', axis=1)
    except Exception as exc:
        print(exc)
        print("May be an etf where this isnt applicable")
        return
    data = data.set_index('Ratios').transpose().reset_index()
    data['month'] = data['index'].str.slice(5)
    data['year'] = data['index'].str.slice(0, 4)
    data['tick'] = tick
    data = data.drop('index', axis=1)
    data = data.set_index(['tick', 'year', 'month'])
    data.columns = map_columns("fin_ratios", list(data.columns.values))

    try:
        url = DATA_MAP['fin_ratios'][1][1].format(tick)
        raw = requests.get(url).content
        data_cm = pd.DataFrame(json.loads(raw)['metrics'])
        data_cm['year'] = data_cm['date'].str.slice(0, 4)
        data_cm['month'] = data_cm['date'].str.slice(5, 7)
        data_cm['tick'] = tick
        data_cm = data_cm.drop('date', axis=1)
        data_cm = data_cm.set_index(['tick', 'year', 'month'])
        data_cm.columns = map_columns("fin_ratios", list(data_cm.columns.values))
    except Exception as exc:
        print(exc)
        print("May be an etf where this isnt applicable")
        return
    data_cm = data_cm.drop(['curr_ratio', 'roe', 'debt_to_equity', 'pb_ratio',
                            'pfcf_ratio', 'pe_ratio', 'div_yield'], axis=1)
    data = pd.merge(data, data_cm, left_index=True, right_index=True)
    send_to_db(data, 'fin_ratios', ['tick', 'year', 'month'])


def bal_sheet_api(tick):
    """
    reach out to the balance sheet API on FMP
    """
    url = DATA_MAP['bal_sheet'][1].format(tick)
    try:
        raw = requests.get(url).content
        data = json.loads(raw)['financials']
        data = pd.DataFrame(data)
    except pd.errors.EmptyDataError as exc:
        print(exc)
        print("May be an etf where this isnt applicable")
        raise
    data['month'] = data['date'].str.slice(5)
    data['year'] = data['date'].str.slice(0, 4)
    data = data.drop('date', axis=1)
    data['tick'] = tick
    data = data.set_index(['tick', 'year', 'month'])
    data.columns = map_columns("bal_sheet", list(data.columns.values))
    send_to_db(data, 'bal_sheet', ['tick', 'year', 'month'])


def inc_statement_api(tick):
    """
    reach out to the incone statement API on FMP
    """
    url = DATA_MAP['inc_statement'][1].format(tick)
    try:
        raw = requests.get(url).content
        data = json.loads(raw)['financials']
        data = pd.DataFrame(data)
    except pd.errors.EmptyDataError as exc:
        print(exc)
        print("May be an etf where this isnt applicable")
        raise
    data['month'] = data['date'].str.slice(5)
    data['year'] = data['date'].str.slice(0, 4)
    data = data.drop('date', axis=1)
    data['tick'] = tick
    data = data.set_index(['tick', 'year', 'month'])
    data.columns = map_columns("inc_statement", list(data.columns.values))
    send_to_db(data, 'inc_statement', ['tick', 'year', 'month'])


def cf_statement_api(tick):
    """
    reach out to the Cash Flow statement API on FMP
    """
    url = DATA_MAP['cf_statement'][1].format(tick)
    try:
        raw = requests.get(url).content
        data = json.loads(raw)['financials']
        data = pd.DataFrame(data)
    except pd.errors.EmptyDataError as exc:
        print(exc)
        print("May be an etf where this isnt applicable")
        raise
    data['month'] = data['date'].str.slice(5)
    data['year'] = data['date'].str.slice(0, 4)
    data = data.drop('date', axis=1)
    data['tick'] = tick
    data = data.set_index(['tick', 'year', 'month'])
    data.columns = map_columns("cf_statement", list(data.columns.values))
    send_to_db(data, 'cf_statement', ['tick', 'year', 'month'])


def send_to_db(data_df, table, prim_keys):
    """
    Send the info to the mysql db
    """
    with DBHelper() as dbh:
        dbh.connect()
        for _, vals in data_df.reset_index().iterrows():
            val_dict = vals.to_dict()
            dbh.upsert(table, val_dict, prim_keys)


def send_px_ret_to_db(ticks=None):
    """
    Send the price of financial statement and returns / fwd returns for that day
    need fwd returns as input for ml algos
    """
    if not ticks:
        ticks = []
        stock_file_name = 'iex_available_{}_2019_05_29.txt'
        with open(FILE_PATH + stock_file_name.format('cs'), "r") as file:
            for line in file:
                ticks.append(line.strip())
    # px_df = get_db_pxs(ticks)
    
    count = 1
    print("{} stocks to load".format(len(ticks)))
    empties = []
    already_done = True
    for ind_t in ticks:
        if ind_t == 'CTRL':
            already_done = False
        if already_done:
            count += 1
            continue
        print("calcs for {}".format(ind_t))
        px_df = get_db_pxs([ind_t])
        fr_df = get_ticker_table_data([ind_t], 'fin_ratios')
        if px_df.empty or fr_df.empty:
            print("empty df for {}\n".format(ind_t))
            empties.append(ind_t)
            count += 1
            continue
        ret_table = add_px_ret_to_fr(px_df, fr_df)
        # pdb.set_trace()
        send_to_db(ret_table, 'fin_ratios', ['tick', 'year', 'month'])
        print("loaded {} stocks, just loaded {}".format(count, ind_t))
        count += 1
    print(empties)


DATA_MAP = {
    'fin_ratios': [fin_ratios_api, ["https://financialmodelingprep.com/api/financial-ratios/{}?datatype=csv",
                                    "https://financialmodelingprep.com/api/v3/company-key-metrics/{}?datatype=json"]],
    'bal_sheet': [bal_sheet_api, "https://financialmodelingprep.com/api/v2/financials/balance-sheet-statement/{}?datatype=json"],
    'inc_statement': [inc_statement_api, "https://financialmodelingprep.com/api/v2/financials/income-statement/{}?datatype=json"],
    'cf_statement': [cf_statement_api, "https://financialmodelingprep.com/api/v2/financials/cash-flow-statement/{}?datatype=json"]
}


if __name__ == "__main__":
    # get_available_ticks()
    # get_fmp_data('fin_ratios')
    # get_fmp_data('fin_ratios', ['AAPL'])
    # get_fmp_data('bal_sheet', ['AAPL'])
    # get_fmp_data('bal_sheet')
    # get_fmp_data('inc_statement', ['AAPL'])
    # get_fmp_data('inc_statement')
    # get_fmp_data('cf_statement', ['AAPL'])
    # get_fmp_data('cf_statement')
    send_px_ret_to_db()
