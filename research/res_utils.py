import sys
import pdb, time, requests, csv
import numpy as np
import pandas as pd
import datetime as dt
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
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
margin_cols = ['cogs', 'rd', 'sga', 'netInterestOtherMargin']
bal_sheet_cols = ['totalLiabilities', 'totalCurrentLiabilities', 'totalCurrentAssets', 'accountsRecievable', 'inventory',
                 'accountsPayable', 'cashAndShortTermInv', 'netPPE', 'totalEquity']
gross_cols = ['workingCapital', 'totalAssets', 'enterpriseValue', 'EBITDA', 'nopat', 'dividendPerShare', 'capSpending', 'operCF', 'FCF',
              'totalEquity']

# output cols
int_liq = ['workingCapital', 'tradeWorkingCapital', 'currentRatio', 'quickRatio', 'workingCap_v_Sales']
op_eff = ['receivablesTurnover', 'receivablesDaysOutstanding', 'totalAssetTurnover', 'inventoryTurnover',
          'inventoryDaysOutstanding', 'daysSalesOutstanding', 'equityTurnover', 'payablesTurnover',
          'payablesDaysOutstanding', 'cashConversionCycle']
marg_rats = ['grossMargin', 'operMargin', 'pretaxMargin', 'netMargin', 'EBITDAMargin', 'EBITDA_v_EV', 'EV_v_EBITDA']
ret_rats = ['ROIC', 'RTC', 'ROA', 'ROE', 'ROE_dupont']
risk_anal = ['operLev', 'intCov', 'debtToEquity', 'debtToCap']
cf_anal = ['operCF', 'FCF', 'FCF_min_wc', 'FCF_min_twc', 'retEarnRatio', 'divPayoutRatio', 'constGrowthRate']
pe = ['PE_low_hist', 'PE_high_hist', 'PE_avg_hist', 'PE_curr_hist', 'PE_fwd']
ps = ['PS', 'PS_curr', 'PS_fwd', 'PS_5yr_avg_hist']
pb = ['PB', 'PB_curr', 'PB_fwd', 'PB_5yr_avg_hist']
pcf = ['PCF', 'PCF_curr', 'PCF_fwd', 'PCF_5yr_avg_hist']
pfcf = ['PFCF', 'PFCF_curr', 'PFCF_fwd', 'PFCF_5yr_avg_hist']
peg = ['PEG', 'PEG_5yr_avg', 'PEGY', 'PEGY_5yr_avg', 'divYield']
dfcf = ['constGrowthRate', 'proj_calc_g', '1st_5yr_lt_g', '2nd_5yr_lt_g']

# valuation
valuations = {
    'Hist Comps': ['PE', 'PS', 'PB', 'PCF', 'PFCF'],
    'DFCF': ['2stage', '3stage', 'Component Anal'],
    'PDV': ['pdv_PE_avg_hist', 'pdv_PS', 'pdv_PB', 'pdv_PCF']
}


def makeAPICall(ticker, sheet='bs', per=3, col=10, num=3):
    # Use this for quarterly info
    # Period can be 3 or 12 for quarterly vs annual
    # Sheet can be bs = balance sheet, is = income statement, cf = cash flow statement
    # Column year can be 5 or 10, doesnt really work
    # 1 = None 2 = Thousands 3 = Millions 4 = Billions
    url = 'http://financials.morningstar.com/ajax/ReportProcess4CSV.html?t={0}&reportType={1}&period={2}&dataType=A&order=asc&columnYear={3}&number={4}'.format(ticker, sheet, per, col, num)
    urlData = requests.get(url).content.decode('utf-8')
    if 'Error reading' in urlData or urlData=='':
        # try one more time
        time.sleep(3)
        urlData = requests.get(url).content.decode('utf-8')
    if 'Error reading' in urlData or urlData=='':
        print('API issue - Error - ' + ticker)
        return []
        
    cr = csv.reader(urlData.splitlines(), delimiter=',')
    data = []
    for row in list(cr):
        data.append(row)
    
    if len(data) == 0:
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
            headers = is_replacements(headers)
        elif sheet == 'bs':
            columns = BS_COLS
            # headers = is_replacements(headers)
        elif sheet == 'cf':
            columns = CF_COLS
        data = [[columns[h]] + d[1:] for h, d in zip(headers, data)]
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Error in dictionary setup: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
        # pdb.set_trace()
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
    

def is_replacements(headers):
    first = False
    new_headers = []
    for h in headers:
        if not first:
            if h == 'Basic':
                new_headers.append('EPS' + h)
            else:
                if h == 'Diluted':
                    new_headers.append('EPS' + h)
                    first = True
                else:
                    new_headers.append(h)
        else:
            if h == 'Basic':
                new_headers.append('Shares' + h)
            else:
                if h == 'Diluted':
                    new_headers.append('Shares' + h)
                else:
                    new_headers.append(h)
    return new_headers


def getNextQuarter(index):
    tick = index[1]
    y = int(index[0])
    m = int(index[2].replace("E","")) + 3
    if m > 12:
        m -= 12
        y += 1
    if m < 10:
        m = "0" + str(m)
    return [str(y), tick, str(m)+"E"]


def getNextYear(index):
    return [str(int(index[0])+1), index[1], str(int(float(str(index[2]).replace("E",""))))+"E"]


def get_ticker_info(ticks, table, idx=None, dates=None):
    # Temp to make testing quicker
    t0 = time.time()
    # tickers = pd.read_csv('/home/ubuntu/workspace/ml_dev_work/utils/dow_ticks.csv', header=None)
    with DBHelper() as db:
        db.connect()
        # df = db.select('morningstar', where = 'date in ("2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016")')
        lis = ''
        for t in ticks:
            lis += "'" + t + "', "
        df = db.select(table, where = 'tick in (' + lis[:-2] + ')')
        
    # Getting Dataframe
    t1 = time.time()
    # print("Done Retrieving data, took {0} seconds".format(t1-t0))
    if idx:
        return df.set_index(idx)
    else:
        return df


def dataCumColumns(data):
    tick = data.reset_index()['tick'][0]
    try:
        data = data.drop([('TTM', '', tick)])
    except:
        # no TTM included
        pass
    H1 = data.iloc[-4] + data.iloc[-3]
    M9 = H1 + data.iloc[-2]
    Y1 = M9 + data.iloc[-1]
    data = data.reset_index()
    data.loc[len(data)] = ['H1', tick, ''] + list(H1.values)
    data.loc[len(data)] = ['M9', tick, ''] + list(M9.values)
    data.loc[len(data)] = ['Y1', tick, ''] + list(Y1.values)
    data = data.reindex([0,1,2,5,3,6,4,7])
    data = data.set_index(IDX)
    return data


def removeEmptyCols(data):
    for key in data.keys():
        data[key] = data[key].dropna(axis='columns', how='all')
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
    for y in years:
        if y == min(years):
            last_y = y
            continue
        year_df = df_y[(df_y.year==y[:4]) & (df_y.month==y[4:])].drop(IDX, axis=1).values
        last_y_df = df_y[(df_y.year==last_y[:4]) & (df_y.month==last_y[4:])].drop(IDX, axis=1).values
        yoy = (year_df / last_y_df - 1) * 100
        yoy[abs(yoy) == np.inf] = 0
        where_are_NaNs = np.isnan(yoy)
        yoy[where_are_NaNs] = 0
        data = list(df_y[(df_y.year==y[:4]) & (df_y.month==y[4:])].iloc[0][IDX]) + list(yoy[0])
        df_chg.loc[len(df_chg)] = data
        last_y = y
    
    # need this to add year over year for single year model
    yoy = (df_y.drop(IDX, axis=1).loc[len(df_y)-1].values / df_y.drop(IDX, axis=1).loc[0].values - 1) * 100
    yoy[abs(yoy) == np.inf] = 0
    where_are_NaNs = np.isnan(yoy)
    yoy[where_are_NaNs] = 0
    data = ['YoY', data_df.reset_index().tick[0], ''] + list(yoy)
    df_chg.loc[len(df_chg)] = data
    df_chg = df_chg.set_index(IDX)
    return df_chg


def setup_comp_cols(indices):
    cols = ['ticker', 'cat'] + [i[0] for i in indices]
    cols.insert(7, 'avg_5y')
    return cols
    
    
def setup_pdv_cols():
    return ['ticker', 'cat', '5y_avg', 'hist_var_v_weight_avg', '2yr_fwd_mult', 'cur_var_v_weight_avg', 'prem_disc', 'pdv_price']


