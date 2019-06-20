"""
Helper script for equity valuation
"""
import pdb
import sys
import time
import csv
import datetime as dt
import requests
import numpy as np
import pandas as pd

from utils.db_utils import DBHelper

IDX = ['year', 'tick', 'month']


# input cols
OLS_COLS = ['total_liabs', 'total_cur_liabs', 'total_cur_assets', 'cash'
            'receivables', 'inv', 'accounts_payable', 'net_ppe', 'total_equity',
            'weight_avg_shares', 'working_cap', 'total_assets', 'ebitda',
            'enterprise_val', 'nopat', 'div_per_share', 'cap_exp', 'oper_cf',
            'revenue', 'oper_inc', 'prov_inc_tax', 'fcf', 'cogs', 'rnd', 'sga',
            'net_int_inc', 'fcf_min_twc', 'gross_prof_marg', 'pretax_prof_marg',
            'net_prof_marg']

# valuation
VALUATIONS = {
    'Hist Comps': ['PE', 'PS', 'PB', 'PCF', 'PFCF'],
    'DFCF': ['2stage', '3stage', 'Component Anal'],
    'PDV': ['pdv_pe_avg_hist', 'pdv_ps_ratio', 'pdv_pd_ratio', 'pdv_pfcc_ratio']
}


def make_api_call(ticker, sheet='bs', per=3, col=10, num=3):
    """
    *** Morngingstar API no longer functional ***
    Call Morningstar API for quarterly data
    Use this for quarterly info
    Period can be 3 or 12 for quarterly vs annual
    Sheet: bs = balance sheet, is = income statement, cf = cash flow statement
    Column year can be 5 or 10, doesnt really work
    1 = None 2 = Thousands 3 = Millions 4 = Billions
    """
    url = 'http://financials.morningstar.com/ajax/ReportProcess4CSV.html?t={0}&reportType={1}&period={2}&dataType=A&order=asc&columnYear={3}&number={4}'.format(ticker, sheet, per, col, num)
    url_data = requests.get(url).content.decode('utf-8')
    if 'Error reading' in url_data or url_data == '':
        # try one more time
        time.sleep(3)
        url_data = requests.get(url).content.decode('utf-8')
    if 'Error reading' in url_data or url_data == '':
        print('API issue - Error - ' + ticker)
        return []

    cr_data = csv.reader(url_data.splitlines(), delimiter=',')
    data = []
    for row in list(cr_data):
        data.append(row)

    if data:
        print('API issue - empty')
        return []

    # Remove empty rows
    data = [x for x in data if x != []]
    data = [x for x in data if len(x) != 1]
    # Remove dates
    dates = data[0][1:]
    data = data[1:]
    # Remove certain strings from headers (USD, Mil, etc) and removing first 2 lines
    headers = [d[0] for d in data]
    # Replace headers with better col names
    try:
        if sheet == 'is':
            columns = []
            # columns = IS_COLS
            # headers = is_replacements(headers)
        elif sheet == 'bs':
            columns = []
            # columns = BS_COLS
            # headers = is_replacements(headers)
        elif sheet == 'cf':
            columns = []
            # columns = CF_COLS
        data = [[columns[h]] + d[1:] for h, d in zip(headers, data)]
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Error in dictionary setup: {0}, {1}, {2}"
              "".format(exc_type, exc_tb.tb_lineno, exc_obj))
        raise

    data = pd.DataFrame(data).transpose()
    cols = data.iloc[0]
    data = data[1:]
    data = data.apply(pd.to_numeric)
    data.columns = cols
    years = [d.split('-')[0] for d in dates]
    if sheet != 'bs':
        months = [d.split('-')[1] for d in dates[:-1]+["-"]]
    else:
        months = [d.split('-')[1] for d in dates]
    data['date'] = years
    data['month'] = months
    data['ticker'] = ticker
    # Need this to fill in columns not included
    for i in columns.values():
        if i not in data.columns:
            data[i] = 0
    data = data.set_index(IDX)
    return data


def get_next_quarter(index):
    """
    Get next quarters index key
    """
    tick = index[1]
    yrs = int(index[0])
    mon = int(index[2].replace("E", "")) + 3
    if mon > 12:
        mon -= 12
        yrs += 1
    if mon < 10:
        mon = "0" + str(mon)
    return [str(yrs), tick, str(mon)+"E"]


def get_next_year(index):
    """
    Get next years index key
    """
    return [str(int(index[0])+1), index[1], str(int(float(str(index[2]).replace("E", ""))))+"E"]


def get_ticker_info(ticks, table, idx=None):
    """
    Grabbing info for a list of given tickers from the db
    """
    # t0 = time.time()
    with DBHelper() as dbh:
        dbh.connect()
        lis = ''
        for tick in ticks:
            lis += "'" + tick + "', "
        df_ret = dbh.select(table, where='tick in (' + lis[:-2] + ')')

    # t1 = time.time()
    # print("Done Retrieving data, took {0} seconds".format(t1-t0))
    if idx:
        return df_ret.set_index(idx)
    return df_ret


def data_cum_columns(data):
    """
    cumulative columns for intrayear calcs
    """
    tick = data.reset_index()['tick'][0]
    try:
        data = data.drop([('TTM', '', tick)])
    except KeyError:
        # no TTM included
        pass
    h1_dat = data.iloc[-4] + data.iloc[-3]
    m9_dat = h1_dat + data.iloc[-2]
    y1_dat = m9_dat + data.iloc[-1]
    data = data.reset_index()
    data.loc[len(data)] = ['H1', tick, ''] + list(h1_dat.values)
    data.loc[len(data)] = ['M9', tick, ''] + list(m9_dat.values)
    data.loc[len(data)] = ['Y1', tick, ''] + list(y1_dat.values)
    data = data.reindex([0, 1, 2, 5, 3, 6, 4, 7])
    data = data.set_index(IDX)
    return data


def remove_empty_cols(data):
    """
    remove empty columns from dataframe
    """
    for key in data.keys():
        data[key] = data[key].dropna(axis='columns', how='all')
    return data


def replace_needed_cols(data):
    """
    replace columns we need with 0s
    """
    replace_checks = []
    replace_checks.append(['bs', 'inv'])
    replace_checks.append(['cf', 'divs_paid'])
    replace_checks.append(['bs', 'short_term_debt'])
    replace_checks.append(['bs', 'long_term_debt'])
    replace_checks.append(['bs', 'accounts_payable'])
    replace_checks.append(['is', 'ebitda'])
    replace_checks.append(['cf', 'chg_working_cap'])
    for ind_rc in replace_checks:
        if not ind_rc[1] in data[ind_rc[0]].columns:
            data[ind_rc[0]][ind_rc[1]] = 0
    return data


def period_chg(data_df):
    """
    Calculate the change fom one year to the next
    """
    data_df = data_df['is']
    df_y = data_df.reset_index()
    df_y = df_y[df_y.year != 'TTM']
    years = list(df_y['year'] + df_y['month'])
    df_chg = pd.DataFrame(columns=IDX + list(data_df.columns))
    for yrs in years:
        if yrs == min(years):
            last_y = yrs
            continue
        year_df = df_y[(df_y.year == yrs[:4]) &
                       (df_y.month == yrs[4:])].drop(IDX, axis=1).values
        last_y_df = df_y[(df_y.year == last_y[:4]) &
                         (df_y.month == last_y[4:])].drop(IDX, axis=1).values
        yoy = (year_df / last_y_df - 1) * 100
        yoy[abs(yoy) == np.inf] = 0
        where_are_nans = np.isnan(yoy)
        yoy[where_are_nans] = 0
        data = list(df_y[(df_y.year == yrs[:4]) &
                         (df_y.month == yrs[4:])].iloc[0][IDX]) + list(yoy[0])
        df_chg.loc[len(df_chg)] = data
        last_y = yrs

    # need this to add year over year for single year model
    yoy = (df_y.drop(IDX, axis=1).loc[len(df_y)-1].values
           / df_y.drop(IDX, axis=1).loc[0].values - 1) * 100
    yoy[abs(yoy) == np.inf] = 0
    where_are_nans = np.isnan(yoy)
    yoy[where_are_nans] = 0
    data = ['YoY', data_df.reset_index().tick[0], ''] + list(yoy)
    df_chg.loc[len(df_chg)] = data
    df_chg = df_chg.set_index(IDX)
    return df_chg


def setup_comp_cols(indices):
    """
    Sets up the comparison analysis columns
    """
    cols = ['ticker', 'cat'] + [i[0] for i in indices]
    cols.insert(7, 'avg_5y')
    return cols


def setup_pdv_cols(per, years_fwd):
    """
    Assigns the column values for the peer derived value calc
    """
    cols = ['ticker', 'cat', '5y_avg', 'hist_avg_v_weight_avg']
    for yrf in range(1, years_fwd + 1):
        year = int(per[0]) + yrf
        cols += ['fwd_mult_{}'.format(year), 'fwd_mult_v_weight_avg_{}'.format(year),
                 'prem_disc_{}'.format(year), 'pdv_price_{}'.format(year)]
    return cols


def match_px(data, eod_px, tick):
    """
    Apply a price to each statement date
    """
    dates = data['bs'].reset_index()[['year', 'month']]
    eod_px = eod_px.loc[tick]
    data['ols']['date_px'] = None
    data['ols']['hi_52wk'] = None
    data['ols']['lo_52wk'] = None
    data['ols']['avg_52wk'] = None

    for _, vals in dates.iterrows():
        # get the closest price to the data date
        data_date = dt.datetime(int(vals['year']), int(vals['month']), 1)
        yr1_ago = dt.datetime(int(vals['year'])-1, int(vals['month']), 1)
        day = 1
        while True:
            try:
                date = dt.datetime(int(vals['year']), int(vals['month']), day)
                px_val = eod_px.loc[date]['px']
                data['ols'].at[(vals['year'], tick, vals['month']), 'date_px'] = px_val
                break
            except KeyError:
                # holiday or weekend probably
                day += 1
            if day > 10:
                break
        # 52 week high, low, and avg
        date_range = eod_px.loc[yr1_ago: data_date]
        data['ols'].at[(vals['year'], tick, vals['month']), 'hi_52wk'] = date_range['px'].max()
        data['ols'].at[(vals['year'], tick, vals['month']), 'lo_52wk'] = date_range['px'].min()
        data['ols'].at[(vals['year'], tick, vals['month']), 'avg_52wk'] = date_range['px'].mean()
    return data


def get_beta(data, eod_px, ticker, mkt, ind):
    """
    Calculate the beta for a given security
    """
    window = 52
    # This will get the week end price and do a pct change
    tick = eod_px.loc[ticker].rename(columns={'px': ticker}).groupby(pd.TimeGrouper('W')).nth(0).pct_change()
    ind = eod_px.loc[ind].rename(columns={'px': ind}).groupby(pd.TimeGrouper('W')).nth(0).pct_change()
    mkt = eod_px.loc[mkt].rename(columns={'px': mkt}).groupby(pd.TimeGrouper('W')).nth(0).pct_change()
    cov_df = pd.merge(tick, mkt, left_index=True, right_index=True).rolling(window, min_periods=1).cov()
    cov_df = cov_df[[cov_df.columns[1]]]
    covariance = cov_df[np.in1d(cov_df.index.get_level_values(1), [ticker])]
    variance_mkt = cov_df[np.in1d(cov_df.index.get_level_values(1), [mkt.columns[0]])]
    beta = (covariance.reset_index().set_index('date')[[mkt.columns[0]]]
            / variance_mkt.reset_index().set_index('date')[[mkt.columns[0]]])
    rep_dates = get_report_dates(data)
    data['ols']['beta'] = None
    for ind_dt in rep_dates:
        try:
            val = beta[beta.index < ind_dt].iloc[-1].values[0]
        except IndexError:
            # print("May not have any dates, assume a beta of 1")
            val = 1
        data['ols'].at[(str(ind_dt.year), ticker, ind_dt.strftime("%m")), 'beta'] = val
        # data['ols'].at[(str(ind_dt.year), ticker, "0" + str(ind_dt.month)), 'beta'] = val
    return data


def get_report_dates(data):
    """
    Gets the report dates from the financial statements
    """
    dates = []
    for ind, _ in data['bs'].iterrows():
        dates.append(dt.datetime(int(ind[0]), int(ind[2]), 1))
    return dates
