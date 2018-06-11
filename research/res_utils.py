import sys
import pdb, time, requests, csv
import numpy as np
import pandas as pd
import datetime as dt

idx = ['date', 'ticker', 'month']

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
    data = data.set_index(idx)
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
