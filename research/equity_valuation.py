import sys
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import pdb, time, requests, csv
import numpy as np
import pandas as pd
import datetime as dt
import quandl
quandl.ApiConfig.api_key = 'J4d6zKiPjebay-zW7T8X'
import pandas_datareader as dr

from res_utils import *
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


def peer_derived_value(data, comp_anal, period):
    # get group values first
    group_vals = {}
    vals = ['PS', 'PE_avg_hist', 'PB', 'PCF']
    ticks = list(data.keys())
    
    # get group market cap
    group_mkt_cap = 0
    for k,v in data.items():
        per = tuple([period[0], k, data[k][0].index.values[0][2]])
        group_mkt_cap += v[0]['shares'][per] * v[0]['currentPrice'][per]
    for v in vals:
        group_vals[v] = 0
        group_vals[v+"_w_avg"] = 0
        for t in ticks:
            per = tuple([period[0], t, data[t][0].index.values[0][2]])
            # 5yr avgs
            group_vals[v] += data[t][0][v].rolling(center=False,window=5).mean()[per] / len(ticks)
            group_vals[v+"_w_avg"] += data[t][0][v].rolling(center=False,window=5).mean()[per] * ((data[t][0]['shares'][per] * data[t][0]['currentPrice'][per]) / group_mkt_cap)
    print(group_vals)
    pdb.set_trace()
    print()
    


def comparison_anal(data, period):
    comp_df = pd.DataFrame()
    years_back = 5
    years_fwd = 2
    cols = ['netIncome', 'revenue', 'grossMargin', 'pretaxMargin', 'netMargin', 'PE_avg_hist']
    cagr = ['netIncome', 'revenue']
    for tick, df in data.items():
        indices = [d for d in list(df[0].index.values) if int(d[0]) > int(period[0]) - years_back and int(d[0]) <= int(period[0])+years_fwd]
        df = df[0].ix[indices]
        if comp_df.empty:
            comp_df = pd.DataFrame(columns=setup_comp_cols(indices))
        for cc in cols:
            if cc in cagr:
                # cagr = compound annual growth rate
                avg = (df[cc][years_back-1] / df[cc][0])**(1/5) - 1
                row = [tick, cc] + list(df[cc].values)
                row.insert(2+years_back, avg)
                comp_df.loc[len(comp_df)] = row
                
                # standard avg of growth rate
                avg = df[cc].pct_change()[1:years_back].mean()
                row = [tick, cc+"_g"] + list(df[cc].pct_change().values)
                row.insert(2+years_back, avg)
                comp_df.loc[len(comp_df)] = row
            else:
                # standard avg
                avg = df[cc][:years_back].mean()
                row = [tick, cc] + list(df[cc].values)
                row.insert(2+years_back, avg)
                comp_df.loc[len(comp_df)] = row
    return comp_df.set_index(['ticker', 'cat'])


def price_perf_anal(period, mkt, ind, hist_px):
    px_df = pd.DataFrame()
    mkt_px = hist_px[mkt].reset_index()
    ind_px = hist_px[ind].reset_index()
    for t in hist_px.columns:
        t_df = pd.DataFrame([t], columns=['tick'])
        t_px = hist_px[t].dropna().reset_index()
        t_df['cur_px'] = t_px[t].values[-1]
        t_df['y_px'] = t_px[t_px['date'] >= dt.date(int(period[0]),1,1).strftime("%Y-%m-%d")][t].values[0]
        t_df['ytd_chg'] = (t_df['cur_px'] / t_df['y_px']) - 1
        t_df['ytd_high'] = max(t_px[t_px['date'] >= dt.date(int(period[0]),1,1).strftime("%Y-%m-%d")][t].values)
        t_df['ytd_low'] = min(t_px[t_px['date'] >= dt.date(int(period[0]),1,1).strftime("%Y-%m-%d")][t].values)
        t_df['mkt_rel_perf'] = t_df['ytd_chg'] - (mkt_px[mkt].values[-1] / mkt_px[mkt_px['date'] >= dt.date(int(period[0]),1,1).strftime("%Y-%m-%d")][mkt].values[0] - 1)
        t_df['ind_rel_perf'] = t_df['ytd_chg'] - (ind_px[ind].values[-1] / ind_px[ind_px['date'] >= dt.date(int(period[0]),1,1).strftime("%Y-%m-%d")][ind].values[0] - 1)
        px_df = px_df.append(t_df)
    return px_df.set_index('tick')    
    

def discount_fcf(data, period, ests):
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
    wacc = (cost_equity * eq_v_cap) + (cost_debt * debt_v_cap) * (1 - tax_rate / 100)
    
    eps_g_proj = 0.12 # analysts projected EPS growth
    data['proj_calc_g'] = (data['constGrowthRate'] + eps_g_proj) / 2 # average of calc'd growth and analyst projection
    data['1st_5yr_lt_g'] = data['constGrowthRate'].rolling(center=False, window=5).mean() # 5yr avg of constant growth calc
    data['2nd_5yr_lt_g'] = data['1st_5yr_lt_g'] - 0.02 # slightly lower than 1st growth calc, usually 2 to 4%
    term_growth = 0.05 # long term terminal growth rate - typically some average of gdp and the industry standard
    
    # 2 Stage DFCF
    years_to_terminal = 2
    FCF_pershare = data['FCF_min_twc'][period] / data['shares'][period]
    indices = [d for d in list(data.index.values) if int(d[0]) > int(period[0]) and int(d[0]) <= int(period[0])+years_to_terminal]
    FCFs = [FCF_pershare * (1 + data['1st_5yr_lt_g'][period])**(int(x[0])-int(period[0])) for x in indices]
    disc_FCFs = [FCFs[x] / (1+cost_equity)**(int(indices[x][0])-int(period[0])) for x in range(0,len(indices))]
    sum_of_disc_CF = sum(disc_FCFs)
    term_val = (data['FCF_min_twc'][indices[-1]] / data['shares'][period]) / (cost_equity - term_growth)
    final_val = term_val + sum_of_disc_CF
    ests.append(("2stage", indices[-1][1], indices[-1][0], '%.3f'%(final_val)))
    print("2 Stage Val Est {} {}: {}".format(indices[-1][1], indices[-1][0], '%.3f'%(final_val)))
    
    # 3 Stage DFCF
    years_phase1 = 1
    years_phase2 = 1
    # 1st growth phase
    FCF_pershare = data['FCF_min_twc'][period] / data['shares'][period]
    indices = [d for d in list(data.index.values) if int(d[0]) > int(period[0]) and int(d[0]) <= int(period[0])+years_phase1]
    FCFs = [FCF_pershare * (1 + data['1st_5yr_lt_g'][period])**(int(x[0])-int(period[0])) for x in indices]
    disc_FCFs_1 = [FCFs[x] / (1+cost_equity)**(int(indices[x][0])-int(period[0])) for x in range(0,len(indices))]
    # second growth phase
    indices = [d for d in list(data.index.values) if int(d[0]) > int(indices[-1][0]) and int(d[0]) <= int(indices[-1][0])+years_phase2]
    FCFs = [FCF_pershare * (1 + data['2nd_5yr_lt_g'][period])**(int(x[0])-int(period[0])) for x in indices]
    disc_FCFs_2 = [FCFs[x] / (1+cost_equity)**(int(indices[x][0])-int(period[0])) for x in range(0,len(indices))]
    sum_of_disc_CF = sum(disc_FCFs_1) + sum(disc_FCFs_2)
    term_val = (data['FCF_min_twc'][indices[-1]] / data['shares'][period]) / (cost_equity - term_growth)
    final_val = term_val + sum_of_disc_CF
    ests.append(("3stage", indices[-1][1], indices[-1][0], '%.3f'%(final_val)))
    print("3 Stage Val Est {} {}: {}".format(indices[-1][1], indices[-1][0], '%.3f'%(final_val)))
    
    # Component DFCF
    years_to_terminal = 2
    # use the OLS growth calcs for FCFs
    FCFs = data['FCF_min_twc'] / data['shares'][period]
    indices = [d for d in list(data.index.values) if int(d[0]) > int(period[0]) and int(d[0]) <= int(period[0])+years_to_terminal]
    disc_FCFs = [FCFs[indices[x]] / (1+cost_equity)**(int(indices[x][0])-int(period[0])) for x in range(0,len(indices))]
    sum_of_disc_CF = sum(disc_FCFs)
    term_val = (data['FCF_min_twc'][indices[-1]] / data['shares'][period]) / (cost_equity - term_growth)
    final_val = term_val + sum_of_disc_CF
    ests.append(("Component Anal", indices[-1][1], indices[-1][0], '%.3f'%(final_val)))
    print("Component Val {} {}: {}".format(indices[-1][1], indices[-1][0], '%.3f'%(final_val)))

    return data, ests


def historical_ratios(data, period):
    ests = []
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
        final_val = '%.3f'%(data['PE_5yr_avg_hist'][period] * data['EPS'][p])
        ests.append(("PE", p[1], p[0], final_val))
        print("Hist avg PE: {}  Fwd EPS: {}  DV Est {} {}: {}".format('%.3f'%(data['PE_5yr_avg_hist'][period]), 
            '%.3f'%(data['EPS'][p]), p[1], p[0], final_val))
    
    # P/S
    data['PS'] = (data['52WeekAvg'] * data['shares']) / data['revenue']
    data['PS_curr'] = (data['currentPrice'] * data['shares']) / data['revenue']
    data['PS_fwd'] = (data['currentPrice'] * data['shares']) / data['revenue'].shift(1)
    data['PS_5yr_avg_hist'] = data['PS'].rolling(center=False,window=5).mean()
    for p in [next_per, pers_2]: 
        final_val = '%.3f'%(data['PS_5yr_avg_hist'][period] * data['revenue'][p] / data['shares'][period])
        ests.append(("PS", p[1], p[0], final_val))
        print("Hist avg PS: {}  Fwd Rev/share: {}  DV Est {} {}: {}".format('%.3f'%(data['PS_5yr_avg_hist'][period]), 
            '%.3f'%(data['revenue'][p] / data['shares'][period]), p[1], p[0], final_val))
    
    # P/B
    data['PB'] = (data['52WeekAvg'] * data['shares']) / data['totalEquity']
    data['PB_curr'] = (data['currentPrice'] * data['shares']) / data['totalEquity']
    data['PB_fwd'] = (data['currentPrice'] * data['shares']) / data['totalEquity'].shift(1)
    data['PB_5yr_avg_hist'] = data['PB'].rolling(center=False,window=5).mean()
    for p in [next_per, pers_2]:
        final_val = '%.3f'%(data['PB_5yr_avg_hist'][period] * data['totalEquity'][p] / data['shares'][period])
        ests.append(("PB", p[1], p[0], final_val))
        print("Hist avg PB: {}  Fwd equity/share: {}  DV Est {} {}: {}".format('%.3f'%(data['PB_5yr_avg_hist'][period]), 
            '%.3f'%(data['totalEquity'][p] / data['shares'][period]), p[1], p[0], final_val))
    
    # P/CF
    data['PCF'] = (data['52WeekAvg'] * data['shares']) / data['operCF']
    data['PCF_curr'] = (data['currentPrice'] * data['shares']) / data['operCF']
    data['PCF_fwd'] = (data['currentPrice'] * data['shares']) / data['operCF'].shift(1)
    data['PCF_5yr_avg_hist'] = data['PCF'].rolling(center=False,window=5).mean()
    for p in [next_per, pers_2]:
        final_val = '%.3f'%(data['PCF_5yr_avg_hist'][period] * data['operCF'][p] / data['shares'][period])
        ests.append(("PCF", p[1], p[0], final_val))
        print("Hist avg PCF: {}  Fwd CF/share: {}  DV Est {} {}: {}".format('%.3f'%(data['PCF_5yr_avg_hist'][period]), 
            '%.3f'%(data['operCF'][p] / data['shares'][period]), p[1], p[0], final_val))
    
    # P/FCF
    data['PFCF'] = (data['52WeekAvg'] * data['shares']) / data['FCF']
    data['PFCF_curr'] = (data['currentPrice'] * data['shares']) / data['FCF']
    data['PFCF_fwd'] = (data['currentPrice'] * data['shares']) / data['FCF'].shift(1)
    data['PFCF_5yr_avg_hist'] = data['PFCF'].rolling(center=False,window=5).mean()
    for p in [next_per, pers_2]:
        final_val = '%.3f'%(data['PFCF_5yr_avg_hist'][period] * data['FCF'][p] / data['shares'][period])
        ests.append(("PFCF", p[1], p[0], final_val))
        print("Hist avg PFCF: {}  Fwd FCF/share: {}  DV Est {} {}: {}".format('%.3f'%(data['PFCF_5yr_avg_hist'][period]), 
            '%.3f'%(data['FCF'][p] / data['shares'][period]), p[1], p[0], final_val))
    
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
    return data, ests
    

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
    data['constGrowthRate'] = data['ROE'] * data['retEarnRatio']
    return data


def model_est_ols(years, cum, hist, chg, margin, avg_cols=[], use_last=[]):
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


def model_est(cum, hist, chg, margin):
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


def margin_df(df):
    data = df.apply(lambda x: x / x['revenue'], axis=1)
    data['taxes'] = df['taxes'] / df['EBT']     # tax rate presented as percentage of pre tax income
    return data


def ols_calc(xs, ys, n_idx=None):
    xs = [dt.datetime(int(x[0]), int(float(str(x[1]).replace("E",""))), 1).date() for x in xs.values]
    start = xs[0]
    xs = get_year_deltas(xs)
    A = np.vstack([xs, np.ones(len(xs))]).T
    slope, yint = np.linalg.lstsq(A, ys.values)[0]
    # new_x = get_year_deltas([start, dt.datetime(int(n_idx[0]), int(n_idx[2][:-1]), 1).date()])[-1]
    return (slope, yint)


def hist_adjustments(hist, data_is):
    # assume sum of last 4 quarters as annual
    hist['depAndAmort'] = data_is['depAndAmort'].mean() * 4
    hist['EBITDA'] = data_is['EBITDA'][-4:].mean() * 4
    
    # Adding columns
    # Net Operatin Profat after tax
    hist['nopat'] = hist['operatingIncome'] * (1 - hist['taxRate']/100)
    hist['operCF'] = hist['netIncome'] + hist['depAndAmort']
    hist['FCF'] = hist['operCF'] - hist['capSpending']
    return hist


def analyze_ests(data, period, ests):
    print("Tick: {}   Date: {} {}".format(period[1], period[0], period[2]))
    print("Current Price: {}".format(data['currentPrice'][period]))
    est_dict = {}
    for e in ests:
        try:
            est_dict[(e[1], e[2])].append(e[3])
        except:
            est_dict[(e[1], e[2])] = [e[3]]
        print("Model: {}  tick: {}  year: {}  EST: {}".format(e[0], e[1], e[2], e[3]))
    
    for k,v in est_dict.items():
        v = [float(vs) for vs in v]
        avg_est = sum(v) / len(v)
        print("tick: {}  year: {} # Models: {}   AVG EST: {}".format(k[0], k[1], str(len(v)), '%.3f'%avg_est))
        
        prem_disc = (avg_est / data['currentPrice'][period]) - 1
        # Divide by beta
        risk_adj = ((avg_est / data['currentPrice'][period]) - 1) / data['beta'][period] 
        print("Prem/Disc: {}  Risk Adj Prem/Disc: {}".format('%.4f'%prem_disc, '%.4f'%risk_adj))


def get_price_data(ticks, comps, period, ests, api=False):
    pxs = pd.DataFrame()
    # Cant find an API I can trust for EOD stock data
    for t in ticks + comps:
        if api:
            start = dt.date(int(data.index.values[0][0]), int(data.index.values[0][2]), 1).strftime("%Y-%m-%d")
            end = dt.datetime.today().date().strftime("%Y-%m-%d")
            try:
                # df = dr.data.DataReader(period[1], 'google', start, end)
                # data = quandl.get_table('WIKI/PRICES', ticker = [period[1]], qopts = { 'columns': ['ticker', 'date', 'adj_close'] }, date = { 'gte': start, 'lte': end }, paginate=True)
                url = "https://www.quandl.com/api/v1/datasets/WIKI/{0}.csv?column=4&sort_order=asc&trim_start={1}&trim_end={2}".format(period[1], start, end)
                qr = pd.read_csv(url)
                qr['Date'] = qr['Date'].astype('datetime64[ns]')
            except:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print("Could not read time series data for {3}: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj, period[1]))
                raise
        else:
            qr = pd.read_csv("/home/ubuntu/workspace/python_for_finance/research/data_grab/{}.csv".format(t))
            qr = qr.rename(columns={'close': t}).set_index(['date'])
            if pxs.empty:
                pxs = qr
            else:
                pxs = pd.merge(pxs, qr, how='left', left_index=True, right_index=True)
    return pxs


def valuation_model(ticks, mode='db'):
    full_data = {}
    for t in ticks:
        if mode == 'api':
            data = makeAPICall(t, 'is')
            # reorganizing columns
            # data = reorgCols(data_cum, data_chg)
        else:
            data = get_ticker_info([t], 'morningstar_monthly_is')
            data_bs = get_ticker_info([t], 'morningstar_monthly_bs')[['totalCash', 'totalLiab']]
            data_hist = get_ticker_info([t], 'morningstar')
            # join on whatever data you need
            data = data.join(data_bs)
    
        data = removeEmptyCols(data)
        data_cum = dataCumColumns(data)
        data_chg = period_chg(data)
        data_margin = margin_df(data)[['grossProfit', 'cogs', 'rd', 'sga', 'mna', 'otherExp']]
        
        # Add some necessary columns and adjustemnts
        data_hist = hist_adjustments(data_hist, data)
        # project out data based on historicals using OLS regression
        # data_est = model_est(data_cum, data_hist, data_chg, data_margin)
        data_ols = model_est_ols(10, data_cum, data_hist, data_chg, data_margin, [], [])
        # Calculate Ratios for Use later
        ratios = ratios_and_valuation(data_ols)
        period = [i for i in ratios.index.values if "E" not in i[2]][-1]
        # Get Historical Ratios for valuation
        hist_rats, ests = historical_ratios(ratios, period)
        # Discounted Free Cash Flow Valuation
        dfcf, ests = discount_fcf(hist_rats, period, ests)
        full_data[t] = [dfcf, ests]
    
    ind = 'XLK'
    mkt = 'SPY'
    other = [mkt, ind]
    # Get Historical Price data
    hist_px = get_price_data(ticks, other, period, ests, False)
    
    # calculate performance metrics based on price
    px_df = price_perf_anal(period, mkt, ind, hist_px)
    
    # Comaprisons
    comp_anal = comparison_anal(full_data, period)
    
    # Peer Derived Value
    pdv = peer_derived_value(full_data, comp_anal, period)
    
    # Analysis of all valuations
    pdb.set_trace()
    analyze_ests(full_data, period)
    

if __name__ == '__main__':
    # income_state_model(['MSFT'], 'api')
    valuation_model(['MSFT', 'AAPL', 'CSCO'])
    