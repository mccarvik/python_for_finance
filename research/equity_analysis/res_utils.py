"""
Helper script for equity valuation
"""
import sys
import time
import csv
import datetime as dt
import requests
import numpy as np
import pandas as pd

from utils.db_utils import DBHelper

IDX = ['year', 'tick', 'month']

IS_COLS = {
    'Revenue' : 'revenue',
    'Total net revenue' : 'revenue',
    'Total revenues' : 'revenue',
    'Revenues, net of interest expense' : 'revNetInt',
    'Cost of revenue' : 'cogs',
    'Gross profit' : 'grossProfit',
    'Research and development' : 'rd',
    'Operating expenses' : 'operatingCost',
    'Sales, General and administrative' : 'sga',
    'Selling, general and administrative' : 'sga',
    'Total operating expenses' : 'operatingCost',
    'Total costs and expenses' : 'operatingCost',
    'Total expenses' : 'operatingCost',
    'Total benefits, claims and expenses' : 'operatingCost',
    'Operating income' : 'EBIT',
    'Interest Expense' : 'intExp',
    'Interest expense' : 'intExp',
    'Interest expenses' : 'intExp',
    'Total interest expense' : 'intExp',
    'Interest income' : 'intInc',
    'Total interest income' : 'intInc',
    'Dividend income' : 'divInc',
    'Total interest and dividend income' : 'intDivInc',
    'Net interest income' : 'netIntInc',
    'Net interest inc after prov for loan losses' : 'netIntInc',
    'Deposits' : 'deposits',
    'Deposits with banks' : 'deposits',
    'Loans and Leases' : 'loanLease',
    'Provision for loan losses' : 'provForLoanLoss',
    'Other assets' : 'otherAssets',
    'Other income (loss)' : 'otherInc',
    'Other income (expense)' : 'otherInc',
    'Other income' : 'otherInc',
    'Other' : 'otherInc',
    'Other expenses' : 'otherExp',
    'Other expense' : 'otherExp',
    'Other operating expenses' : 'otherExp',
    'Nonrecurring expense' : 'nonRecExp',
    'Policy acquisition and other expenses' : 'policyAcqExp',
    'Income before taxes' : 'EBT',
    'Income before income taxes' : 'EBT',
    'Income (loss) from cont ops before taxes' : 'EBT',
    'Provision for income taxes' : 'taxes',
    'Provision (benefit) for taxes' : 'taxes',
    'Provision (benefit) for income taxes' : 'taxes',
    'Income tax (expense) benefit' : 'taxes',
    'Income taxes' : 'taxes',
    'Net income from continuing operations' : 'netIncomeContinuing',
    'Net income from continuing ops' : 'netIncomeContinuing',
    'Total nonoperating income, net' : 'netIncomeNonOperating',
    'Net income from discontinuing ops' : 'netIncomeDiscontinuing',
    'Income from discontinued ops' : 'netIncomeDiscontinuing',
    'Income from discontinued operations' : 'netIncomeDiscontinuing',
    'Net income' : 'netIncome',
    'Net income available to common shareholders' : 'netIncomeShareholders',
    'EPSBasic' : 'EPSBasic',
    'EPSDiluted' : 'EPS',
    'SharesBasic' : 'sharesBasic',
    'SharesDiluted' : 'shares',
    'EBITDA' : 'EBITDA',
    'Restructuring, merger and acquisition' : 'mna',
    'Merger, acquisition and restructuring' : 'mna',
    'Short-term borrowing' : 'shortTermBorrow',
    'Commissions and fees' : 'commAndFees',
    'Lending and deposit-related fees' : 'lendAndDeposit',
    'Securities gains (losses)' : 'secGains',
    'Securities' : 'secGains',
    'Gain on sale of equity investment' : 'gainEquitySale',
    'Credit card income' : 'ccInc',
    'Total noninterest revenue' : 'nonIntRev',
    'Total noninterest expenses' : 'nonIntExp',
    'Total noninterest expense' : 'nonIntExp',
    'Provisions for credit losses' : 'provCredLoss',
    'Compensation and benefits' : 'compBens',
    'Occupancy expense' : 'occExp',
    'Tech, communication and equipment' : 'techCommEq',
    'Technology and occupancy' : 'techCommEq',
    'Professional and outside services' : 'profOutServ',
    'Other special charges' : 'otherSpecCharges',
    'Preferred dividend' : 'prefDividend',
    'Preferred distributions' : 'prefDividend',
    'Preferred dividends' : 'prefDividend',
    'Other distributions' : 'otherDistributions',
    'Minority interest' : 'minorityInt',
    'Depreciation and amortization' : 'depAndAmort',
    'Operation and maintenance' : 'opAndmaintenance',
    'Federal funds sold' : 'fedFundsSold',
    'Federal funds purchased' : 'fedFundsPurchased',
    'Insurance premium' : 'insurancePrem',
    'Insurance commissions' : 'insurancePrem',
    'Amortization of intangibles' : 'amortIntangibles',
    'Premiums' : 'premiums',
    'Service fees and commissions' : 'servFeesAmdComms',
    'Investment income, net' : 'invInc',
    'Realized capital gains (losses), net' : 'capGains',
    'Policyholder benefits and claims incurred' : 'policyBensAndClaims',
    'Interest credited to policyholder accounts' : 'intPolicyAccounts',
    'Policyholder dividends' : 'policyDivs',
    'Advertising and promotion' : 'advertisingProm',
    'Extraordinary items' : 'extraordItems',
    'Borrowed funds' : 'borrowedFunds',
    'Investment banking' : 'invBanking',
    'Asset mgmt and securities services' : 'asstMgmtSecServ',
}

BS_COLS = {
    'Cash and cash equivalents' : 'cash',
    'Restricted cash' : 'restrictedCash',
    'Restricted cash and cash equivalents' : 'restrictedCash',
    'Securities available for sale' : 'secsForSale',
    'Securities held to maturity' : 'secsToMaturity',
    'Securities borrowed' : 'secsBorrowed',
    'Federal Home Loan Bank stock' : 'fhlbStock',
    'Total securities' : 'secsTotal',
    'Short-term investments' : 'shortTermInv',
    'Securities and investments' : 'shortTermInv',
    'Trading securities' : 'shortTermInv',
    'Trading account securities' : 'shortTermInv',
    'Trading assets' : 'shortTermInv',
    'Investments' : 'shortTermInv',
    'Derivative assets' : 'derivAssets',
    'Derivative liabilities' : 'derivLiabs',
    'Trading liabilities' : 'shortTermLiab',
    'Total cash and cash equivalents' : 'totalCash',
    'Total cash' : 'totalCash',
    'Receivables' : 'receivables',
    'Interest receivable' : 'intReceivables',
    'Premiums and other receivables' : 'receivables',
    'Prepaid expenses' : 'prepaidExp',
    'Inventories' : 'inventories',
    'Total current assets' : 'curAssets',
    'Other current assets' : 'otherCurAssets',
    'Gross property, plant and equipment' : 'ppe',
    'Property and equipment, at cost' : 'ppe',
    'Land' : 'land',
    'Other properties' : 'otherProperties',
    'Fixtures and equipment' : 'fixturesEquipment',
    'Premises and equipment' : 'fixturesEquipment',
    'Premises and equipment, net' : 'fixturesEquipment',
    'Accumulated Depreciation' : 'depreciation',
    'Accumulated depreciation' : 'depreciation',
    'Net property, plant and equipment' : 'netPPE',
    'Property, plant and equipment, net' : 'netPPE',
    'Property and equipment' : 'netPPE',
    'Total non-current assets' : 'nonCurAssets',
    'Total assets' : 'assets',
    'Accounts payable' : 'acctsPayable',
    'Payables' : 'acctsPayable',
    'Payables and accrued expenses' : 'acctsPayable',
    'Short-term debt' : 'shortTermDebt',
    'Short-term borrowing' : 'shortTermDebt',
    'Other current liabilities' : 'otherCurLiab',
    'Total current liabilities' : 'totalCurLiab',
    'Long-term debt' : 'longTermDebt',
    'Other long-term assets' : 'otherLongTermAssets',
    'Other long-term liabilities' : 'otherLongTermLiab',
    'Other liabilities' : 'otherLongTermLiab',
    'Total non-current liabilities' : 'totalNonCurLiab',
    'Total liabilities' : 'totalLiab',
    'Accrued liabilities' : 'accruedLiab',
    'Accrued expenses and liabilities' : 'accruedLiab',
    'Deferred taxes liabilities' : 'deferredTaxLiab',
    'Deferred tax liabilities' : 'deferredTaxLiab',
    'Taxes payable' : 'deferredTaxLiab',
    'Income taxes payable' : 'deferredTaxLiab',
    'Deferred taxes' : 'deferredTaxLiab',
    'Deferred income taxes' : 'deferredTaxAsset',
    'Deferred income tax assets' : 'deferredTaxAsset',
    'Deferred revenues' : 'deferredRev',
    'Common stock' : 'commonStock',
    'Retained earnings' : 'retainedEarnings',
    'Goodwill' : 'goodwill',
    'Intangible assets' : 'intangibleAssets',
    'Other intangible assets' : 'otherIntAssets',
    'Other intangibles' : 'otherIntAssets',
    'Other assets' : 'otherAssets',
    'Accumulated other comprehensive income' : 'otherCompInc',
    "Total Stockholders' equity" : 'stockEquity',
    "Total stockholders' equity" : 'stockEquity',
    'Other liabilities and equities' : 'otherLiabAndStockEquity',
    "Total liabilities and stockholders' equity" : 'totalLiabAndStockEquity',
    'Additional paid-in capital' : 'paidInCapital',
    'Treasury stock' : 'tsyStock',
    'Cash and due from banks' : 'cashDueFromBanks',
    'Federal funds sold' : 'fedFundsSold',
    'Federal funds purchased' : 'fedFundsPurchased',
    'Debt securities' : 'debtSecurities',
    'Fixed maturity securities' : 'debtSecurities',
    'Loans' : 'loans',
    'Loans, total' : 'loans',
    'Consumer loans' : 'consumerLoans',
    'Commercial loans' : 'commercialLoans',
    'Residential loans' : 'resiLoans',
    'Total loans' : 'loans',
    'Allowance for loan losses' : 'allowForLoanLoss',
    'Net loans' : 'netLoans',
    'Total loans, net' : 'netLoans',
    'Repurchase agreement' : 'repoAgreement',
    'Non-interest-bearing' : 'depositsNonInt',
    'Interest-bearing' : 'depositsInt',
    'Total deposits' : 'deposits',
    'Deposits' : 'deposits',
    'Deposits with banks' : 'deposits',
    'Equity and other investments' : 'equityAndOtherInv',
    'Equity securities' : 'equityAndOtherInv',
    'Accrued investment income' : 'invInc',
    'Minority interest' : 'minorityInterest',
    'Minority Interest' : 'minorityInterest',
    'Capital leases' : 'capLeases',
    'Pensions and other benefits' : 'pensionsAndOtherBens',
    'Future policy benefits' : 'pensionsAndOtherBens',
    'Pensions and other postretirement benefits' : 'pensionsAndOtherBens',
    'Policyholder funds' : 'policyHolderFunds',
    'Deferred policy acquisition costs' : 'policyAcqCosts',
    'Other Equity' : 'otherEquity',
    'Other equity' : 'otherEquity',
    'Prepaid pension benefit' : 'prepaidPensBen',
    'Prepaid pension costs' : 'prepaidPensBen',
    'Real estate properties' : 'realEstateProps',
    'Foreclosed real estate, net' : 'realEstateProps',
    'Real estate' : 'realEstateProps',
    'Real estate properties, net' : 'realEstatePropsNet',
    'Separate account assets' : 'sepAcctAssets',
    'Separate account liabilities' : 'sepAcctLiab',
    'Unearned premiums' : 'unearnedPrems',
    'Preferred stock' : 'prefStock',
    'Regulatory assets' : 'regulatoryAssets',
    'Regulatory liabilities' : 'regulatoryLiab',
}

CF_COLS = {
    'Net income' : 'netIncome',
    'Free cash flow' : 'freeCashFlow',
    'Provision for credit losses' : 'provCreditLosses',
    'Deferred income taxes' : 'deferredIncTaxes',
    'Deferred tax (benefit) expense' : 'deferredIncTaxes',
    'Income taxes payable' : 'incomeTaxesPayable',
    'Interest payable' : 'intPayable',
    'Cash paid for income taxes' : 'cashPaidTaxes',
    'Cash paid for interest' : 'cashPaidInt',
    'Stock based compensation' : 'stockComp',
    'Inventory' : 'inventory',
    'Investment/asset impairment charges' : 'invImpairmentCharges',
    'Depreciation & amortization' : 'depAndAmort',
    'Amortization of debt discount/premium and issuance costs' : 'amortOfDebtCosts',
    'Amortization of debt and issuance costs' : 'amortOfDebtCosts',
    'Investments (gains) losses' : 'invGainsLosses',
    'Investments losses (gains)' : 'invGainsLosses',
    'Long-term debt issued' : 'longTermDebtIssued',
    'Debt issued' : 'longTermDebtIssued',
    'Long-term debt repayment' : 'longTermDebtRepaid',
    'Debt repayment' : 'longTermDebtRepaid',
    'Accrued liabilities' : 'accruedLiab',
    'Receivable' : 'receivables',
    'Accounts receivable' : 'receivables',
    'Payables' : 'payables',
    'Accounts payable' : 'payables',
    'Prepaid expenses' : 'prepaidExp',
    'Loans issued' : 'loans',
    'Loans' : 'loans',
    'Changes in loans, net' : 'netLoanChg',
    'Common stock issued' : 'commonStockIssued',
    'Common stock repurchased' : 'commonStockRepurch',
    'Cash dividends paid' : 'divsPaid',
    'Dividend paid' : 'divsPaid',
    'Payments from loans' : 'loanPayments',
    'Other assets and liabilities' : 'otherAssetAndLiab',
    'Other operating activities' : 'otherOperActivities',
    'Other investing activities' : 'otherInvActivities',
    'Other financing activities' : 'otherFinActivities',
    'Purchases of investments' : 'invPurchases',
    'Net change in cash' : 'netChgCash',
    'Net cash used for investing activities' : 'netCashInvActs',
    'Net cash provided by (used for) financing activities' : 'netCashFinActs',
    'Net cash provided by operating activities' : 'netCashOperActs',
    'Cash at beginning of period' : 'cashBegPer',
    'Cash at end of period' : 'cashEndPer',
    'Reserves for claims and claim adjustment expenses' : 'reservesForClaims',
    'Sales/maturities of fixed maturity and equity securities' : 'salesAndMaturities',
    'Sales/maturity of investments' : 'salesAndMaturities',
    'Sales/Maturities of investments' : 'salesAndMaturities',
    'Equity in (income) loss from equity method investments' : 'equityMethodInv',
    'Change in working capital' : 'chgWorkingCap',
    'Other working capital' : 'otherWorkingCap',
    'Other non-cash items' : 'otherNonCash',
    'Investments in property, plant, and equipment' : 'invPPE',
    'Property, plant, and equipment reductions' : 'reductPPE',
    'Property, and equipments, net' : 'invPPE',
    'Other investing charges' : 'otherInvCharges',
    'Repurchases of treasury stock' : 'repurchTsyStock',
    'Operating cash flow' : 'operatingCashFlow',
    'Capital expenditure' : 'capExp',
    'Change in deposits' : 'chgDeposits',
    'Change in deposits with banks' : 'chgDeposits',
    'Acquisitions, net' : 'acquisitions',
    'Acquisitions and dispositions' : 'acquisitions',
    'Change in short-term borrowing' : 'chgShortTermBorrow',
    'Short-term borrowing' : 'chgShortTermBorrow',
    'Effect of exchange rate changes' : 'fxRateEffect',
    'Purchases of intangibles' : 'purchaseOfIntangibles',
    'Purchases of intangibles, net' : 'purchaseOfIntangibles',
    'Sales of intangibles' : 'saleOfIntangibles',
    'Excess tax benefit from stock based compensation' : 'excessTaxBenefit',
    'Deferred charges' : 'deferredCharges',
    'Change in federal funds purchased' : 'chgFedFunds',
    'Change in federal funds sold' : 'chgFedFunds',
    'Preferred stock issued' : 'prefStockIssued',
    'Redemption of preferred stock' : 'prefStockRedemption',
    'Preferred stock repaid' : 'prefStockRedemption',
    'Capitalization of deferred policy acquisition costs' : 'capDeferredPolAcq',
    '(Gains) loss on disposition of businesses' : 'gainDispOfBus',
    'Investment income due and accrued' : 'invIncomeDueAndAccrued'
}

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
            columns = IS_COLS
            # headers = is_replacements(headers)
        elif sheet == 'bs':
            columns = BS_COLS
            # headers = is_replacements(headers)
        elif sheet == 'cf':
            columns = CF_COLS
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
    return [str(int(index[0])+1), index[1], str(index[2]).replace("E", "")+"E"]


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
    check_replace = []
    check_replace.append(['bs', 'accounts_payable'])
    check_replace.append(['bs', 'inv'])
    check_replace.append(['cf', 'chg_working_cap'])
    check_replace.append(['cf', 'divs_paid'])
    for ind_check in check_replace:
        if ind_check[1] not in data[ind_check[0]].columns:
            data[ind_check[0]][ind_check[1]] = 0
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
    data['ols']['date_px'] = np.nan
    data['ols']['hi_52wk'] = np.nan
    data['ols']['lo_52wk'] = np.nan
    data['ols']['avg_52wk'] = np.nan

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
    # ind = eod_px.loc[ind].rename(columns={'px': ind}).groupby(pd.TimeGrouper('W')).nth(0).pct_change()
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
            print("May not have any dates, assume a beta of 1")
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
