import sys
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import pdb, time, requests, csv
import numpy as np
import pandas as pd
import datetime as dt

from res_utils import *
from utils.db_utils import DBHelper
from dx.frame import get_year_deltas

# NOTES:
# simulate api call --> http://financials.morningstar.com/ajax/exportKR2CSV.html?t=BA

idx = ['date', 'ticker', 'month']
hist_quarterly_map = {
    'operatingIncome' : 'EBIT',
    'totalLiabilities' : 'totalLiab',
    'cashAndShortTermInv' : 'totalCash',
    'netInterestOtherMargin' : 'otherInc'
}


def discountFCF(data, period):
    cost_debt = 0.07  # should be pulled off debt issued by company, hard coded for now
    
    rf = 0.028 # rate on 10yr tsy
    market_risk_prem = 0.05 # generally accepted market risk premium above risk free rate
    # 1 year beta, take the 3 year average
    beta = data['beta'].rolling(center=False,window=3).mean()[period]
    # CAPM
    cost_equity = rf + beta * market_risk_prem
    
    mv_eq = data['currentPrice'][period] * data['shares'][period]
    # mv_debt = HARD TO GET
    bv_debt = data['shortTermDebt'][period] + data['longTermDebt'][period]
    total_capital = bv_debt + mv_eq
    eq_v_cap = mv_eq / total_capital
    debt_v_cap = bv_debt / total_capital
    tax_rate = data['taxRate'].rolling(center=False,window=5).mean()[period]
    pdb.set_trace()
    wacc = (cost_equity * eq_v_cap) + (cost_debt * debt_v_cap) * (1 - tax_rate / 100)
    


def historical_ratios(data, period):
    next_per = tuple(getNextYear(period))
    pers_2 = tuple(getNextYear(next_per))
    
    # fill current price with latest measurement
    data['currentPrice'] = data['currentPrice'].fillna(data['currentPrice'].dropna()[-1])
    
    # PE Ratios
    data['PE_low_hist'] = (data['52WeekLow'] * data['shares']) / data['netIncome']
    data['PE_high_hist'] = (data['52WeekHigh'] * data['shares']) / data['netIncome']
    data['PE_avg_hist'] = (data['52WeekAvg'] * data['shares']) / data['netIncome']
    data['PE_curr_hist'] = (data['currentPrice'] * data['shares']) / data['netIncome']
    data['PE_fwd'] = (data['currentPrice'] * data['shares']) / data['netIncome'].shift(1)
    data['PE_5yr_avg_hist'] = data['PE_avg_hist'].rolling(center=False,window=5).mean()
    for p in [next_per, pers_2]:
        print("Hist avg PE: {}  Fwd EPS: {}  DV Est {} {}: {}".format('%.3f'%(data['PE_5yr_avg_hist'][period]), 
            '%.3f'%(data['EPS'][p]), p[1], p[0], '%.3f'%(data['PE_5yr_avg_hist'][period] * data['EPS'][p])))
    
    # P/S
    data['PS'] = (data['52WeekAvg'] * data['shares']) / data['revenue']
    data['PS_curr'] = (data['currentPrice'] * data['shares']) / data['revenue']
    data['PS_fwd'] = (data['currentPrice'] * data['shares']) / data['revenue'].shift(1)
    data['PS_5yr_avg_hist'] = data['PS'].rolling(center=False,window=5).mean()
    for p in [next_per, pers_2]:
        print("Hist avg PS: {}  Fwd Rev/share: {}  DV Est {} {}: {}".format('%.3f'%(data['PS_5yr_avg_hist'][period]), 
            '%.3f'%(data['revenue'][p] / data['shares'][period]), p[1], p[0],
            '%.3f'%(data['PS_5yr_avg_hist'][period] * data['revenue'][p] / data['shares'][period])))
    
    # P/B
    data['PB'] = (data['52WeekAvg'] * data['shares']) / data['totalEquity']
    data['PB_curr'] = (data['currentPrice'] * data['shares']) / data['totalEquity']
    data['PB_fwd'] = (data['currentPrice'] * data['shares']) / data['totalEquity'].shift(1)
    data['PB_5yr_avg_hist'] = data['PB'].rolling(center=False,window=5).mean()
    for p in [next_per, pers_2]:
        print("Hist avg PB: {}  Fwd equity/share: {}  DV Est {} {}: {}".format('%.3f'%(data['PB_5yr_avg_hist'][period]), 
            '%.3f'%(data['totalEquity'][p] / data['shares'][period]), p[1], p[0],
            '%.3f'%(data['PB_5yr_avg_hist'][period] * data['totalEquity'][p] / data['shares'][period])))
    
    # P/CF
    data['PCF'] = (data['52WeekAvg'] * data['shares']) / data['operCF']
    data['PCF_curr'] = (data['currentPrice'] * data['shares']) / data['operCF']
    data['PCF_fwd'] = (data['currentPrice'] * data['shares']) / data['operCF'].shift(1)
    data['PCF_5yr_avg_hist'] = data['PCF'].rolling(center=False,window=5).mean()
    for p in [next_per, pers_2]:
        print("Hist avg PCF: {}  Fwd CF/share: {}  DV Est {} {}: {}".format('%.3f'%(data['PCF_5yr_avg_hist'][period]), 
            '%.3f'%(data['operCF'][p] / data['shares'][period]), p[1], p[0],
            '%.3f'%(data['PCF_5yr_avg_hist'][period] * data['operCF'][p] / data['shares'][period])))
    
    # P/FCF
    data['PFCF'] = (data['52WeekAvg'] * data['shares']) / data['FCF']
    data['PFCF_curr'] = (data['currentPrice'] * data['shares']) / data['FCF']
    data['PFCF_fwd'] = (data['currentPrice'] * data['shares']) / data['FCF'].shift(1)
    data['PFCF_5yr_avg_hist'] = data['PFCF'].rolling(center=False,window=5).mean()
    for p in [next_per, pers_2]:
        print("Hist avg PFCF: {}  Fwd FCF/share: {}  DV Est {} {}: {}".format('%.3f'%(data['PFCF_5yr_avg_hist'][period]), 
            '%.3f'%(data['FCF'][p] / data['shares'][period]), p[1], p[0],
            '%.3f'%(data['PFCF_5yr_avg_hist'][period] * data['FCF'][p] / data['shares'][period])))
    
    # Relative P/E
    # NEED THE EARNIGNS OF THE SNP500
    # data['PE_rel'] = (data['52WeekAvg'] * data['shares']) / data['PE_of_SnP']
    # data['PE_rel_curr'] = (data['currentPrice'] * data['shares']) / data['PE_of_SnP']
    # data['PE_rel_fwd'] = (data['currentPrice'] * data['shares']) / data['PE_of_SnP'].shift(1)
    # data['PE_rel__5yr_avg'] = data['PE_rel'].rolling(center=False,window=5).mean()
    # for p in [next_per, pers_2]:
        # print("Hist avg PS: {}  Fwd Rev/share: {}  DV Est {} {}: {}".format(data['PE_rel__5yr_avg'][period], 
        #     data['PE_of_SnP'][p] / data['shares'][period], period[1], period[0]
        #     data['PE_rel__5yr_avg'][period] * data['revenue'][p] / data['shares'][period]))
    
    # PEG
    data['PEG'] = data['PE_avg_hist'] / (data['netIncome'].pct_change() * 100)
    data['PEG_5yr_avg'] = data['PEG'].rolling(center=False,window=5).mean()
    data['divYield'] = data['dividendPerShare'] / data['52WeekAvg']
    data['PEGY'] = data['PE_avg_hist'] / ((data['netIncome'].pct_change() + data['divYield']) * 100)
    data['PEGY_5yr_avg'] = data['PEGY'].rolling(center=False,window=5).mean()
    return data
    

def ratios_and_valuation(data):
    # Internal Liquidity
    data['workingCapital'] = data['totalCurrentAssets'] - data['totalCurrentLiabilities']
    data['tradeWorkingCapital'] = data['accountsRecievable'] + data['inventory'] - data['accountsPayable']
    data['currentRatio'] = data['totalCurrentAssets'] / data['totalCurrentLiabilities']
    data['quickRatio'] = (data['cashAndShortTermInv'] + data['accountsRecievable']) / data['totalCurrentLiabilities']
    data['workingCap_v_Sales'] = data['workingCapital'] / data['revenue']
    
    # Operating Efficiency
    data['receivablesTurnover'] = data['revenue'] / data['accountsRecievable']
    data['receivablesDaysOutstanding'] = 365 / data['receivablesTurnover'] 
    data['totalAssetTurnover'] = data['revenue'] / data['totalAssets']
    data['inventoryTurnover'] = data['cogs'] / data['inventory']
    data['inventoryDaysOutstanding'] = 365 / data['inventoryTurnover'] 
    data['daysSalesOutstanding'] = data['accountsRecievable'] / (data['revenue'] / 365)
    data['fixedAssetTurnover'] = data['revenue'] / data['netPPE']
    data['equityTurnover'] = data['revenue'] / data['totalEquity']
    data['payablesTurnover'] = data['revenue'] / data['accountsPayable']
    data['payablesDaysOutstanding'] = 365 / data['payablesTurnover']
    data['cashConversionCycle'] = data['inventoryDaysOutstanding'] + data['receivablesDaysOutstanding'] - data['payablesDaysOutstanding']
    
    # margin ratios
    data['grossMargin'] = (data['revenue'] - data['cogs']) / data['revenue'] 
    data['operMargin'] = (data['operatingIncome']) / data['revenue']
    data['pretaxMargin'] = data['EBT'] / data['revenue']
    data['netMargin'] = data['netIncome'] / data['revenue']
    data['EBITDAMargin'] = data['EBITDA'] / data['revenue']
    data['EBITDA_v_EV'] = data['EBITDA'] / data['enterpriseValue']
    data['EV_v_EBITDA'] = data['enterpriseValue'] / data['EBITDA']
    
    # return ratios
    data['ROIC'] = data['nopat'] / (data['totalAssets'] - data['cashAndShortTermInv'] - data['totalCurrentLiabilities']) 
    data['RTC'] = data['nopat'] / data['totalAssets']
    data['ROA'] = data['netIncome'] / data['totalAssets']
    data['ROE'] = data['netIncome'] / data['totalEquity']
    data['ROE_dupont'] = (data['netIncome'] / data['revenue']) * (data['revenue'] / data['totalAssets']) * (data['totalAssets'] / data['totalEquity'])
    
    # risk analysis
    data['operLev'] = data['operatingIncome'].pct_change() / data['revenue'].pct_change()
    data['intCov'] = data['operatingIncome'] / data['netInterestOtherMargin']
    data['debtToEquity'] = data['totalLiabilities'] / data['totalEquity']
    data['debtToCap'] = data['totalLiabilities'] / data['totalAssets']
    
    # cash flow analysis
    data['FCF_min_wc'] = data['FCF'] - data['workingCapital'].diff()
    data['FCF_min_twc'] = data['FCF'] - data['tradeWorkingCapital'].diff()
    data['divPayoutRatio'] = (data['dividendPerShare'] * data['shares']) / data['netIncome']
    data['retEarnRatio'] = (1 - data['divPayoutRatio'])
    data['constGrwothRate'] = data['ROE'] * data['retEarnRatio']
    return data


def modelEstOLS(years, cum, hist, chg, margin, avg_cols=[], use_last=[]):
    df_est = pd.DataFrame()
    # some cleanup
    margin = margin.reset_index()[margin.reset_index().date != 'TTM'].set_index(idx)
    cum = cum.reset_index()[cum.reset_index().month != ''].set_index(idx)
    
    # need to convert from margin to gross
    for cc in margin_cols:
        hist[cc] = hist['revenue'] * (hist[cc] / 100)
    for cc in bal_sheet_cols:
        hist[cc] = hist['totalAssets'] * (hist[cc] / 100)
    
    # next qurater est is equal to revenue * average est margin over the last year
    for i in range(years):
        n_idx = list(hist.iloc[-1].name)
        # n_idx = getNextQuarter(n_idx)
        n_idx = getNextYear(n_idx)
        
        n_cum_dict = {k: v for k, v in zip(idx, n_idx)}
        n_hist_dict = {k: v for k, v in zip(idx, n_idx)}
        
        #########
        # Use OLS to get projected values
        #########
        
        for cc in ['revenue', 'operatingIncome', 'taxRate'] + margin_cols + gross_cols + bal_sheet_cols:
            # Need these for columns that are too eradic for OLS
            if cc in avg_cols:
                n_hist_dict[cc] = hist[cc].mean()
                continue
            if cc in use_last:
                n_hist_dict[cc] = hist[cc].values[-1]
                continue
            
            xs = hist.reset_index()[['date','month']]
            slope, yint = ols_calc(xs, hist[cc])
            start = dt.datetime(int(xs.values[0][0]), int(xs.values[0][1]), 1).date()
            new_x = get_year_deltas([start, dt.datetime(int(n_idx[0]), int(n_idx[2][:-1]), 1).date()])[-1]
            # Need this to convert terminology for quarterly, also need to divide by four
            # cc = hist_quarterly_map.get(cc, cc)
            n_hist_dict[cc] = (yint + new_x * slope)
        
        # 0.6 = about corp rate , 0.02 = about yield on cash, 0.25 = 1/4 of the year
        # n_cum_dict['intExp'] = total_debt * 0.25 * 0.7 - cash_and_inv * 0.25 * 0.02
        # n_cum_dict['EBT'] = n_cum_dict['EBIT'] - n_cum_dict['intExp'] + n_cum_dict['otherInc']
        
        # Interest Expense included in other income
        # n_hist_dict['intExp'] = total_debt * 0.06 - cash_and_inv * 0.02 
        n_hist_dict['EBT'] = n_hist_dict['operatingIncome'] + n_hist_dict['netInterestOtherMargin']
        n_hist_dict['taxes'] = (n_hist_dict['taxRate'] / 100) * n_hist_dict['EBT']
        n_hist_dict['netIncome'] = n_hist_dict['EBT'] - n_hist_dict['taxes']
        
        n_hist_dict['shares'] = ((((cum.iloc[-1]['shares'] / cum.iloc[-5]['shares']) - 1) / 4) + 1) * cum.iloc[-1]['shares']
        n_hist_dict['EPS'] = n_hist_dict['netIncome'] / n_hist_dict['shares']
        
        # t_df = pd.DataFrame(n_cum_dict, index=[0]).set_index(idx)
        t_df = pd.DataFrame(n_hist_dict, index=[0]).set_index(idx)
        hist = hist.append(t_df)
    # print(hist[sum_cols])
    return hist


def modelEst(cum, hist, chg, margin):
    df_est = pd.DataFrame()
    # some cleanup
    margin = margin.reset_index()[margin.reset_index().date != 'TTM'].set_index(idx)
    cum = cum.reset_index()[cum.reset_index().month != ''].set_index(idx)
    
    # next qurater est is equal to revenue * average est margin over the last year
    for i in range(4):
        n_idx = list(cum.iloc[-1].name)
        n_idx = getNextQuarter(n_idx)
        n_data = n_idx + list(margin[-5:-1].mean())
        t_df = pd.DataFrame(dict((key, value) for (key, value) in zip(idx+list(margin.columns), n_data)), columns=idx+list(margin.columns), index=[0]).set_index(idx)
        margin = margin.append(t_df)
        
        n_cum_dict = {k: v for k, v in zip(idx, n_idx)}
        n_cum_dict['revenue'] = cum[-5:-1]['revenue'].mean()
        
        #########
        # Use mean of previous few years to get there
        #########
        for c in ['cogs', 'rd', 'sga', 'grossProfit', 'mna', 'otherExp']:
            n_cum_dict[c] = margin.loc[tuple(n_idx)][c] * n_cum_dict['revenue']
        n_cum_dict['operatingCost'] = n_cum_dict['rd'] + n_cum_dict['sga'] + n_cum_dict['mna'] + n_cum_dict['otherExp']
        n_cum_dict['EBIT'] = n_cum_dict['revenue'] - n_cum_dict['operatingCost'] - n_cum_dict['cogs']   # operating income
        # Need to update these when we do balance sheet
        total_debt = cum.iloc[-1]['totalLiab']
        cash_and_inv = cum.iloc[-1]['totalCash']
        # 0.6 = about corp rate , 0.02 = about yield on cash, 0.25 = 1/4 of the year 
        n_cum_dict['intExp'] = total_debt * 0.25 * 0.7 - cash_and_inv * 0.25 * 0.02
        # just assume average of last year, gonna be very specific company to company
        n_cum_dict['otherInc'] = cum[-5:-1]['otherInc'].mean()
        n_cum_dict['EBT'] = n_cum_dict['EBIT'] - n_cum_dict['intExp'] + n_cum_dict['otherInc']
        # average tax rate of the last year
        n_cum_dict['taxes'] = (cum[-5:-1]['taxes'] / cum[-5:-1]['EBT']).mean() * n_cum_dict['EBT']
        n_cum_dict['netIncome'] = n_cum_dict['EBT'] - n_cum_dict['taxes']
        
        # Assume change over the last year continues for next quarter - may need to adjust this per company
        n_cum_dict['shares'] = ((((cum.iloc[-1]['shares'] / cum.iloc[-5]['shares']) - 1) / 4) + 1) * cum.iloc[-1]['shares']
        n_cum_dict['sharesBasic'] = ((((cum.iloc[-1]['sharesBasic'] / cum.iloc[-5]['sharesBasic']) - 1) / 4) + 1) * cum.iloc[-1]['sharesBasic']
        
        n_cum_dict['EPS'] = n_cum_dict['netIncome'] / n_cum_dict['shares']
        n_cum_dict['EPSBasic'] = n_cum_dict['netIncome'] / n_cum_dict['sharesBasic']
        
        # Assume cash and liabilities are static
        n_cum_dict['totalLiab'] = total_debt
        n_cum_dict['totalCash'] = cash_and_inv
        
        t_df = pd.DataFrame(n_cum_dict, index=[0]).set_index(idx)
        cum = cum.append(t_df)
        
    # clean up df
    cum = cum.fillna(0)
    empty_cols = [c for c in list(cum.columns) if all(v == 0 for v in cum[c])]
    cum = cum[list(set(cum.columns) - set(empty_cols))]
    return cum


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


def clean_add_columns(df):
    # Edit the margin based columns
    for c in ['cogs', 'sga', 'rd']:
        df[c] = (df[c] / 100) * df['revenue']
    df['operatingCost'] = df['sga'] + df['rd']
    df['Taxes'] = df['EBT'] - df['netIncome']
    return df
    

def get_ticker_info(ticks, table, dates=None):
    # Temp to make testing quicker
    t0 = time.time()
    # tickers = pd.read_csv('/home/ubuntu/workspace/ml_dev_work/utils/dow_ticks.csv', header=None)
    with DBHelper() as db:
        db.connect()
        # df = db.select('morningstar', where = 'date in ("2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016")')
        lis = ''
        for t in ticks:
            lis += "'" + t + "', "
        df = db.select(table, where = 'ticker in (' + lis[:-2] + ')')
        
    # Getting Dataframe
    t1 = time.time()
    print("Done Retrieving data, took {0} seconds".format(t1-t0))
    return df.set_index(['date', 'ticker', 'month'])


def prune_columns(df):
    desired_cols = ['revenue', 'cogs', 'grossProfit', 'sga', 'rd', 'operatingIncome', 'netInterestOtherMargin', 'other',
                    'EBTMargin', 'taxRate', 'taxesPayable', 'netIncome', 'shares', 'dividendPerShare', 'trailingEPS', 'EBT']
    return df[desired_cols]


def period_chg(df):
    df_y = df.reset_index()
    df_y = df_y[df_y.date != 'TTM']
    years = list(df_y['date'] + df_y['month'])
    # df_y = df_y.set_index(['date', 'ticker', 'month'])
    df_chg = pd.DataFrame(columns=idx + list(df.columns))
    for y in years:
        if y == min(years):
            last_y = y
            continue
        year_df = df_y[(df_y.date==y[:4]) & (df_y.month==y[4:])].drop(idx, axis=1).values
        last_y_df = df_y[(df_y.date==last_y[:4]) & (df_y.month==last_y[4:])].drop(idx, axis=1).values
        yoy = (year_df / last_y_df - 1) * 100
        yoy[abs(yoy) == np.inf] = 0
        where_are_NaNs = np.isnan(yoy)
        yoy[where_are_NaNs] = 0
        data = list(df_y[(df_y.date==y[:4]) & (df_y.month==y[4:])].iloc[0][idx]) + list(yoy[0])
        df_chg.loc[len(df_chg)] = data
        last_y = y
    
    # need this to add year over year for single year model
    yoy = (df_y.drop(idx, axis=1).loc[len(df_y)-1].values / df_y.drop(idx, axis=1).loc[0].values - 1) * 100
    yoy[abs(yoy) == np.inf] = 0
    where_are_NaNs = np.isnan(yoy)
    yoy[where_are_NaNs] = 0
    data = ['YoY', df.reset_index().ticker[0], ''] + list(yoy)
    df_chg.loc[len(df_chg)] = data
    df_chg = df_chg.set_index(idx)
    return df_chg


def margin_df(df):
    data = df.apply(lambda x: x / x['revenue'], axis=1)
    data['taxes'] = df['taxes'] / df['EBT']     # tax rate presented as percentage of pre tax income
    return data


def dataCumColumns(data):
    tick = data.reset_index()['ticker'][0]
    try:
        data = data.drop([('TTM', tick, '')])
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
    data = data.set_index(idx)
    return data



def reorgCols(cums, chgs):
    # just orders the columsn for use, VERY rigid
    return cums.append(chgs).reset_index().reindex([0, 1, 8, 2, 9, 3, 4, 10, 5, 6, 11, 7, 12]).set_index(idx)
    

def ols_calc(xs, ys, n_idx=None):
    xs = [dt.datetime(int(x[0]), int(float(str(x[1]).replace("E",""))), 1).date() for x in xs.values]
    start = xs[0]
    xs = get_year_deltas(xs)
    A = np.vstack([xs, np.ones(len(xs))]).T
    slope, yint = np.linalg.lstsq(A, ys.values)[0]
    # new_x = get_year_deltas([start, dt.datetime(int(n_idx[0]), int(n_idx[2][:-1]), 1).date()])[-1]
    return (slope, yint)


def removeEmptyCols(df):
    return df.dropna(axis='columns', how='all')


def histAdjustments(hist, data_is):
    # assume sum of last 4 quarters as annual
    hist['depAndAmort'] = data_is['depAndAmort'].mean() * 4
    hist['EBITDA'] = data_is['EBITDA'][-4:].mean() * 4
    
    # Adding columns
    # Net Operatin Profat after tax
    hist['nopat'] = hist['operatingIncome'] * (1 - hist['taxRate']/100)
    hist['operCF'] = hist['netIncome'] + hist['depAndAmort']
    hist['FCF'] = hist['operCF'] - hist['capSpending']
    return hist


def income_state_model(ticks, mode='db'):
    if mode == 'api':
        data = makeAPICall(ticks[0], 'is')
        # reorganizing columns
        # data = reorgCols(data_cum, data_chg)
    else:
        data = get_ticker_info(ticks, 'morningstar_monthly_is')
        data_bs = get_ticker_info(ticks, 'morningstar_monthly_bs')[['totalCash', 'totalLiab']]
        data_hist = get_ticker_info(ticks, 'morningstar')
        # join on whatever data you need
        data = data.join(data_bs)

    data = removeEmptyCols(data)
    data_cum = dataCumColumns(data)
    data_chg = period_chg(data)
    data_margin = margin_df(data)[['grossProfit', 'cogs', 'rd', 'sga', 'mna', 'otherExp']]
    
    data_hist = histAdjustments(data_hist, data)
    # data_est = modelEst(data_cum, data_hist, data_chg, data_margin)
    data_ols = modelEstOLS(5, data_cum, data_hist, data_chg, data_margin, [], [])
    ratios = ratios_and_valuation(data_ols)
    period = [i for i in ratios.index.values if "E" not in i[2]][-1]
    hist_rats = historical_ratios(ratios, period)
    dfcf = discountFCF(hist_rats,period)
    
    # data = prune_columns(data)
    # data = clean_add_columns(data)
    # data_chg = period_chg(data)
    # data = data.join(data_chg.set_index(idx), how='inner')
    pdb.set_trace()
    print()


if __name__ == '__main__':
    # income_state_model(['MSFT'], 'api')
    income_state_model(['MSFT'])
    # income_state_model(['MCD'])
    # income_state_model(['CSCO'])
    # income_state_model(['FLWS'], mode='db')
    # income_state_model(['FLWS'], mode='api')
    