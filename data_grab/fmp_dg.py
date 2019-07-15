"""
Script to download varous datasets from finanical modeling prep website
"""
import pdb
import sys
import time
import json
import warnings
import datetime as dt
import requests
import pandas as pd
sys.path.append("/home/ec2-user/environment/python_for_finance/")
from utils.db_utils import DBHelper, get_ticker_table_data
from data_grab.fmp_helper import map_columns, add_px_ret_to_fr, \
                                 add_px_vol_to_fr
from data_grab.eod_px_util import get_db_pxs, FILE_NAME_STOCK

warnings.filterwarnings("ignore")
SUCCESS = []
FAILURE = []

FILE_PATH = '/home/ec2-user/environment/python_for_finance/data_grab/inputs/'
FILE_NAME = 'fmp_avail_symbols_{}.txt'

# fmp info --> https://financialmodelingprep.com/developer/docs#Financial-Ratios


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
        with open(FILE_PATH + FILE_NAME.format("20190715"), "r") as file:
            for line in file:
                tasks.append(line.strip())
    else:
        tasks = tickers
    time0 = time.time()
    print("total stocks: {}".format(len(tasks)))

    count = 0
    already_have = True
    # already_have = False
    for tick in tasks:
        print(tick)
        # if tick == 'RIG':
        #     already_have = False
        # if already_have:
        #     count += 1
        #     print("skipping {}  already have data".format(tick))
        #     continue
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
            # pdb.set_trace()
            print("Failed " + tick + "\t")
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
    data = json.loads(raw)['ratios']
    data_rows = []
    for report in data:
        data_dict = {}
        data_dict['date'] = report['date']
        # loop through sections
        for section in list(report.keys())[1:]:
            for item in report[section].keys():
                data_dict[item] = report[section][item]
        data_rows.append(data_dict)
    data = pd.DataFrame.from_dict(data_rows, orient='columns')

    # data = data.set_index('Ratios').transpose().reset_index()
    data['month'] = data['date'].str.slice(5, 7)
    data['year'] = data['date'].str.slice(0, 4)
    data['tick'] = tick
    data = data.drop('date', axis=1)
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
    data = data.drop(['curr_ratio', 'roe', 'debt_to_equity', 'pb_ratio', 'fcf_per_share',
                      'pfcf_ratio', 'pe_ratio', 'div_yield', 'cash_per_share',
                      'receivables_turnover', 'days_of_inv_on_hand', 'payout_ratio',
                      'days_payables_outstanding', 'inv_turnover', 'ps_ratio',
                      'days_sales_outstanding', 'oper_cf_per_share', 'pocf_ratio',
                      'payables_turnover', 'receivables_turnover'], axis=1)
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
    data['month'] = data['date'].str.slice(5, 7)
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
    data['month'] = data['date'].str.slice(5, 7)
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
        while True:
            if "Bad Gateway" in str(raw):
                print("{} Bad Gateway - sleeping for 1 second".format(tick))
                time.sleep(1)
                raw = requests.get(url).content
            else:
                break

        data = json.loads(raw)['financials']
        data = pd.DataFrame(data)
        if data.empty:
            print("No data from API for {}".format(tick))
            return
    except pd.errors.EmptyDataError as exc:
        print(exc)
        print("May be an etf where this isnt applicable")
        raise
    data['month'] = data['date'].str.slice(5, 7)
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
        with open(FILE_PATH + FILE_NAME_STOCK.format("20190715"), "r") as file:
            for line in file:
                ticks.append(line.strip())
    # px_df = get_db_pxs(ticks)

    count = 0
    batch_size = 100
    print("{} stocks to load".format(len(ticks)))
    empties = []
    already_done = True
    
    # Get all of the financial Ratios table
    fr_tot = get_ticker_table_data(ticks, 'fin_ratios').set_index('tick')
    
    while count + batch_size < len(ticks):
        batch_ticks = ticks[count:count+batch_size]
        # batch_ticks = ['A', 'AA', 'AAPL']
        px_tot = get_db_pxs(batch_ticks).reset_index().set_index('tick')
    
        for ind_t in batch_ticks:
            # if ind_t == 'SLP':
            #     already_done = False
            # if already_done:
            #     count += 1
            #     continue
            # pdb.set_trace()
            print("calcs for {}".format(ind_t))
            try:
                fr_df = fr_tot.loc[ind_t]
                px_df = px_tot.loc[ind_t].reset_index()[['date','px']].set_index('date')
                # px_df = get_db_pxs([ind_t]).reset_index().set_index('tick')
            except KeyError:
                print("no prices for {}\n".format(ind_t))
                empties.append(ind_t)
                count += 1
                continue
            
            if px_df.empty or fr_df.empty:
                print("empty df for {}\n".format(ind_t))
                empties.append(ind_t)
                count += 1
                continue
            try:
                ret_table = add_px_ret_to_fr(px_df, fr_df.reset_index())
                ret_table = add_px_vol_to_fr(px_df, ret_table.reset_index())
                send_to_db(ret_table, 'fin_ratios', ['tick', 'year', 'month'])
                print("loaded {} stocks, just loaded {}".format(count, ind_t))
                count += 1
            except KeyError:
                print("May have just one year for {}\n".format(ind_t))
                empties.append(ind_t)
                count += 1
                continue
        print("Missed tickers: {}".format(empties))


DATA_MAP = {
    'fin_ratios': [fin_ratios_api, ["https://financialmodelingprep.com/api/v3/financial-ratios/{}?datatype=csv",
                                    "https://financialmodelingprep.com/api/v3/company-key-metrics/{}?datatype=json"]],
    'bal_sheet': [bal_sheet_api, "https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/{}?datatype=json"],
    'inc_statement': [inc_statement_api, "https://financialmodelingprep.com/api/v3/financials/income-statement/{}?datatype=json"],
    'cf_statement': [cf_statement_api, "https://financialmodelingprep.com/api/v3/financials/cash-flow-statement/{}?datatype=json"]
}


if __name__ == "__main__":
    get_available_ticks()
    get_fmp_data('bal_sheet', ['OBCI'])
    # get_fmp_data('bal_sheet')
    get_fmp_data('inc_statement', ['OBCI'])
    # get_fmp_data('inc_statement')
    get_fmp_data('fin_ratios', ['OBCI'])
    # get_fmp_data('fin_ratios')
    get_fmp_data('cf_statement', ['OBCI'])
    # get_fmp_data('cf_statement')
    send_px_ret_to_db(['OBCI'])
