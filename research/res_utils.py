import sys
import pdb, time, requests, csv
import numpy as np
import pandas as pd
import datetime as dt

idx = ['date', 'ticker', 'month']

IS_COLS = {
    'Revenue' : 'revenue',
    'Total net revenue' : 'revenue',
    'Cost of revenue' : 'cogs',
    'Gross profit' : 'grossProfit',
    'Research and development' : 'rd',
    'Operating expenses' : 'operatingCost',
    'Sales, General and administrative' : 'sga',
    'Total operating expenses' : 'operatingCost', 
    'Operating income' : 'EBIT', 
    'Interest Expense' : 'intExp',
    'Interest expense' : 'intExp',
    'Total interest expense' : 'intExp',
    'Interest income' : 'intInc',
    'Total interest income' : 'intInc',
    'Net interest income' : 'netIntInc',
    'Deposits' : 'deposits',
    'Loans and Leases' : 'loanLease',
    'Other assets' : 'otherAssets',
    'Other income (expense)' : 'otherInc', 
    'Other income' : 'otherInc',
    'Other' : 'otherInc',
    'Other expenses' : 'otherExp',
    'Other expense' : 'otherExp',
    'Other operating expenses' : 'otherExp',
    'Income before taxes' : 'EBT',
    'Income before income taxes' : 'EBT',
    'Income (loss) from cont ops before taxes' : 'EBT',
    'Provision for income taxes' : 'taxes', 
    'Provision (benefit) for taxes' : 'taxes',
    'Net income from continuing operations' : 'netIncomeContinuing', 
    'Net income from discontinuing ops' : 'netIncomeDiscontinuing',
    'Net income' : 'netIncome', 
    'Net income available to common shareholders' : 'netIncomeShareholders', 
    'EPSBasic' : 'EPSBasic', 
    'EPSDiluted' : 'EPS', 
    'SharesBasic' : 'sharesBasic', 
    'SharesDiluted' : 'shares', 
    'EBITDA' : 'EBITDA',
    'Restructuring, merger and acquisition' : 'mna',
    'Short-term borrowing' : 'shortTermBorrow',
    'Commissions and fees' : 'commAndFees',
    'Lending and deposit-related fees' : 'lendAndDeposit',
    'Securities gains (losses)' : 'secGains',
    'Credit card income' : 'ccInc',
    'Total noninterest revenue' : 'nonIntRev',
    'Total noninterest expenses' : 'nonIntExp',
    'Provisions for credit losses' : 'provCredLoss',
    'Compensation and benefits' : 'compBens',
    'Occupancy expense' : 'occExp',
    'Tech, communication and equipment' : 'techCommEq',
    'Professional and outside services' : 'profOutServ',
    'Other special charges' : 'otherSpecCharges',
    'Preferred dividend' : 'prefDividend',
    'Minority interest' : 'minorityInt',
    'Depreciation and amortization' : 'depAndAmort',
}




# Provision (benefit) for taxes




def makeAPICall(ticker, sheet='bs', per=3, col=10, num=3):
    # Use this for quarterly info
    # Period can be 3 or 12 for quarterly vs annual
    # Sheet can be bs = balance sheet, is = income statement, cf = cash flow statement
    # Column year can be 5 or 10, doesnt really work
    # 1 = None 2 = Thousands 3 = Millions 4 = Billions
    url = 'http://financials.morningstar.com/ajax/ReportProcess4CSV.html?t={0}&reportType={1}&period={2}&dataType=A&order=asc&columnYear={3}&number={4}'.format(ticker, sheet, per, col, num)
    print(url)
    urlData = requests.get(url).content.decode('utf-8')
    if 'Error reading' in urlData or urlData=='':
        # try one more time
        time.sleep(5)
        urlData = requests.get(url).content.decode('utf-8')
    if 'Error reading' in urlData or urlData=='':
        print('API issue - Error')
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
            data = [[columns[h]] + d[1:] for h, d in zip(headers, data)]
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Error in dictionary setup: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
        pdb.set_trace()
        print()
    
    data = pd.DataFrame(data).transpose()
    cols = data.iloc[0]
    data = data[1:]
    data = data.apply(pd.to_numeric)
    data.columns = cols
    years = [d.split('-')[0] for d in dates]
    months = [d.split('-')[1] for d in dates[:-1]+["-"]]
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
