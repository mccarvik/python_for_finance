"""
Script to perform equity research and analysis as described in
Equity Valuation for Analysts and Investors by James Kelleher
"""
import sys
import pdb
import datetime as dt
import numpy as np
import pandas as pd
import quandl
quandl.ApiConfig.api_key = 'J4d6zKiPjebay-zW7T8X'

sys.path.append("/home/ec2-user/environment/python_for_finance/")
from res_utils import get_ticker_info, removeEmptyCols, period_chg, \
                      getNextYear, OLS_COLS, match_px, getNextQuarter
from dx.frame import get_year_deltas

# NOTES:
# simulate api call --> http://financials.morningstar.com/ajax/exportKR2CSV.html?t=BA

IDX = ['year', 'tick', 'month']
DEBUG = True


def peer_derived_value(data, comp_anal, period, hist_px):
    """
    Get the value of the stock as compared to its peers
    """
    # get group values first
    group_vals = {}
    years_fwd = 2
    vals = ['PS', 'PE_avg_hist', 'PB', 'PCF']
    ticks = list(data.keys())

    # get group market cap
    group_mkt_cap = 0
    for k, v in data.items():
        per = tuple([period[0], k, data[k][0].index.values[0][2]])
        try:
            group_mkt_cap += v[0]['shares'][per] * hist_px[k].dropna().values[-1]
        except:
            # May not have reported yet for this year, if this also fails,
            # raise exception as may have bigger problem
            per = tuple([str(int(period[0])-1), k, data[k][0].index.values[0][2]])
            group_mkt_cap += v[0]['shares'][per] * float(hist_px[k].dropna().values[-1])
            per = tuple([period[0], k, data[k][0].index.values[0][2]])
        # Need to project out vals
        for val in vals:
            xs = data[k][0][val].dropna().reset_index()[['date', 'month']]
            slope, yint = ols_calc(xs, data[k][0][val].dropna())
            for fwd in range(1, years_fwd+1):
                start = dt.datetime(int(xs.values[0][0]), int(xs.values[0][1]), 1).date()
                per = tuple([str(int(period[0])+fwd), v[0].index.values[0][1], v[0].index.values[0][2]+"E"])
                new_x = get_year_deltas([start, dt.datetime(int(per[0]), int(per[2][:-1]), 1).date()])[-1]
                data[k][0].at[per, val] = (yint + new_x * slope)

    for v in vals:
        group_vals[v] = 0
        group_vals[v+"_w_avg"] = 0
        group_vals[v+"_fwd"] = 0
        group_vals[v+"_fwd_w_avg"] = 0
        for t in ticks:
            per = tuple([period[0], t, data[t][0].index.values[0][2]])
            fwd = tuple([str(int(period[0])+years_fwd), t, data[t][0].index.values[0][2]+"E"])

            # 5yr avgs, simple and weighted
            try:
                group_vals[v] += data[t][0][v].rolling(center=False, window=5).mean()[per] / len(ticks)
                group_vals[v+"_w_avg"] += data[t][0][v].rolling(center=False, window=5).mean()[per] * ((data[t][0]['shares'][per] * float(hist_px[t].values[-1])) / group_mkt_cap)
            except:
                # May not have reported yet for this year, if this also fails, raise exception as may have bigger problem
                per = tuple([str(int(period[0])-1), t, data[t][0].index.values[0][2]])
                group_vals[v] += data[t][0][v].rolling(center=False, window=5).mean()[per] / len(ticks)
                group_vals[v+"_w_avg"] += data[t][0][v].rolling(center=False, window=5).mean()[per] * ((data[t][0]['shares'][per] * float(hist_px[t].values[-1])) / group_mkt_cap)

            # 5yr avgs, simpel and weighted
            group_vals[v+"_fwd"] += data[t][0][v].rolling(center=False, window=years_fwd).mean()[fwd] / len(ticks)
            group_vals[v+"_fwd_w_avg"] += data[t][0][v].rolling(center=False, window=years_fwd).mean()[fwd] * ((data[t][0]['shares'][per] * float(hist_px[t].values[-1])) / group_mkt_cap)
        if DEBUG:
            print("{} 5Y simple avg: {}".format(v, '%.3f'%group_vals[v]))
            print("{} 5Y weighted avg: {}".format(v, '%.3f'%group_vals[v+"_w_avg"]))
            print("{} 2Y fwd avg: {}".format(v, '%.3f'%group_vals[v+"_fwd"]))
            print("{} 2Y fwd weighted avg: {}".format(v, '%.3f'%group_vals[v+"_fwd_w_avg"]))
    comp_df = pd.DataFrame()
    for k, v in data.items():
        per = tuple([period[0], k, data[k][0].index.values[0][2]])
        for val in vals:
            if comp_df.empty:
                comp_df = pd.DataFrame(columns=setup_pdv_cols())
            row = [k, val]
            try:
                row.append(v[0][val].rolling(center=False, window=5).mean()[per])
            except:
                # May not have reported yet for this year, if this also fails, raise exception as may have bigger problem
                per = tuple([str(int(period[0])-1), k, data[k][0].index.values[0][2]])
                row.append(v[0][val].rolling(center=False, window=5).mean()[per])
            row.append(row[-1] / group_vals[val+"_w_avg"])
            # end of fwd estimate year
            fwd = tuple([str(int(period[0])+years_fwd), k, data[k][0].index.values[0][2]+"E"])
            row.append(v[0][val].rolling(center=False, window=years_fwd).mean()[fwd])
            row.append(row[-1] / group_vals[val+"_fwd_w_avg"])
            row.append(row[3] / row[5])
            row.append(float(hist_px[k].dropna().values[-1]) * row[-1])
            data[k][1].append(tuple(["pdv_"+val, per[1], str(int(per[0])+years_fwd), '%.3f'%row[-1]]))
            comp_df.loc[len(comp_df)] = row
    return data, comp_df.set_index(['ticker', 'cat'])


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
    if DEBUG:
        print(comp_df.set_index(['ticker', 'cat']))
    return comp_df.set_index(['ticker', 'cat'])


def price_perf_anal(period, mkt, ind, hist_px):
    px_df = pd.DataFrame()
    mkt_px = hist_px[mkt].reset_index()
    ind_px = hist_px[ind].reset_index()
    for t in hist_px.columns:
        t_df = pd.DataFrame([t], columns=['tick'])
        t_px = hist_px[t].dropna().reset_index()
        t_df['cur_px'] = float(t_px[t].values[-1])
        t_df['y_px'] = float(t_px[t_px['date'] >= dt.date(int(period[0]), 1, 1).strftime("%Y-%m-%d")][t].values[0])
        t_df['ytd_chg'] = (t_df['cur_px'] / t_df['y_px']) - 1
        t_df['ytd_high'] = max(t_px[t_px['date'] >= dt.date(int(period[0]), 1, 1).strftime("%Y-%m-%d")][t].values)
        t_df['ytd_low'] = min(t_px[t_px['date'] >= dt.date(int(period[0]), 1, 1).strftime("%Y-%m-%d")][t].values)
        t_df['mkt_rel_perf'] = t_df['ytd_chg'] - (mkt_px[mkt].values[-1] / mkt_px[mkt_px['date'] >= dt.date(int(period[0]), 1, 1).strftime("%Y-%m-%d")][mkt].values[0] - 1)
        t_df['ind_rel_perf'] = t_df['ytd_chg'] - (ind_px[ind].values[-1] / ind_px[ind_px['date'] >= dt.date(int(period[0]), 1, 1).strftime("%Y-%m-%d")][ind].values[0] - 1)
        px_df = px_df.append(t_df)
    return px_df.set_index('tick')


def discount_fcf(data, period, ests, hist_px):
    cost_debt = 0.07  # should be pulled off debt issued by company, hard coded for now

    rf = 0.028 # rate on 10yr tsy
    market_risk_prem = 0.05 # generally accepted market risk premium above risk free rate
    # 1 year beta, take the 3 year average
    beta = data['beta'].rolling(center=False, window=3).mean()[period]
    # CAPM
    cost_equity = rf + beta * market_risk_prem

    mv_eq = float(hist_px[period[1]].values[-1]) * data['shares'][period]
    # mv_debt = HARD TO GET
    bv_debt = data['shortTermDebt'][period] + data['longTermDebt'][period]
    total_capital = bv_debt + mv_eq
    eq_v_cap = mv_eq / total_capital
    debt_v_cap = bv_debt / total_capital
    tax_rate = data['taxRate'].rolling(center=False, window=5).mean()[period]
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
    disc_FCFs = [FCFs[x] / (1+cost_equity)**(int(indices[x][0])-int(period[0])) for x in range(0, len(indices))]
    sum_of_disc_CF = sum(disc_FCFs)
    term_val = (data['FCF_min_twc'][indices[-1]] / data['shares'][period]) / (cost_equity - term_growth)
    final_val = term_val + sum_of_disc_CF
    ests.append(("2stage", indices[-1][1], indices[-1][0], '%.3f'%(final_val)))
    if DEBUG:
        print("2 Stage Val Est {} {}: {}".format(indices[-1][1], indices[-1][0], '%.3f'%(final_val)))

    # 3 Stage DFCF
    years_phase1 = 1
    years_phase2 = 1
    # 1st growth phase
    FCF_pershare = data['FCF_min_twc'][period] / data['shares'][period]
    indices = [d for d in list(data.index.values) if int(d[0]) > int(period[0]) and int(d[0]) <= int(period[0])+years_phase1]
    FCFs = [FCF_pershare * (1 + data['1st_5yr_lt_g'][period])**(int(x[0])-int(period[0])) for x in indices]
    disc_FCFs_1 = [FCFs[x] / (1+cost_equity)**(int(indices[x][0])-int(period[0])) for x in range(0, len(indices))]
    # second growth phase
    indices = [d for d in list(data.index.values) if int(d[0]) > int(indices[-1][0]) and int(d[0]) <= int(indices[-1][0])+years_phase2]
    FCFs = [FCF_pershare * (1 + data['2nd_5yr_lt_g'][period])**(int(x[0])-int(period[0])) for x in indices]
    disc_FCFs_2 = [FCFs[x] / (1+cost_equity)**(int(indices[x][0])-int(period[0])) for x in range(0, len(indices))]
    sum_of_disc_CF = sum(disc_FCFs_1) + sum(disc_FCFs_2)
    term_val = (data['FCF_min_twc'][indices[-1]] / data['shares'][period]) / (cost_equity - term_growth)
    final_val = term_val + sum_of_disc_CF
    ests.append(("3stage", indices[-1][1], indices[-1][0], '%.3f'%(final_val)))
    if DEBUG:
        print("3 Stage Val Est {} {}: {}".format(indices[-1][1], indices[-1][0], '%.3f'%(final_val)))

    # Component DFCF
    years_to_terminal = 2
    # use the OLS growth calcs for FCFs
    FCFs = data['FCF_min_twc'] / data['shares'][period]
    indices = [d for d in list(data.index.values) if int(d[0]) > int(period[0]) and int(d[0]) <= int(period[0])+years_to_terminal]
    disc_FCFs = [FCFs[indices[x]] / (1+cost_equity)**(int(indices[x][0])-int(period[0])) for x in range(0, len(indices))]
    sum_of_disc_CF = sum(disc_FCFs)
    term_val = (data['FCF_min_twc'][indices[-1]] / data['shares'][period]) / (cost_equity - term_growth)
    final_val = term_val + sum_of_disc_CF
    ests.append(("Component Anal", indices[-1][1], indices[-1][0], '%.3f'%(final_val)))
    if DEBUG:
        print("Component Val {} {}: {}".format(indices[-1][1], indices[-1][0], '%.3f'%(final_val)))

    return data, ests


def historical_ratios(data, period, hist_px):
    """
    Calculate historical ratios for valuation
    """
    ests = []
    next_per = tuple(getNextYear(period))
    pers_2 = tuple(getNextYear(next_per))

    # fill current price with latest measurement
    curr_px = hist_px.loc[period[1]].iloc[-1]['px']

    # PE Ratios
    net_inc = pd.concat([data['is']['net_inc'], data['ols']['net_inc'].dropna()])
    data['ols']['eps'] = net_inc / data['ols']['weight_avg_shares']
    data['ols']['pe_low_hist'] = data['ols']['lo_52wk'] / data['ols']['eps']
    data['ols']['pe_low_hist'] = data['ols']['hi_52wk'] / data['ols']['eps']
    data['ols']['pe_avg_hist'] = data['ols']['avg_52wk'] / data['ols']['eps']
    data['ols']['pe_curr_hist'] = curr_px / data['ols']['eps']
    data['ols']['pe_fwd'] = ((data['ols']['date_px'] * data['is']['weight_avg_shares'])
                             / data['is']['net_inc'].shift(1))
    data['ols']['pe_5yr_avg_hist'] = data['ols']['pe_avg_hist'].rolling(center=False, window=5).mean()

    for per in [next_per, pers_2]:
        final_val = '%.3f' % (data['ols']['pe_5yr_avg_hist'][period] * (data['ols']['eps'][per]))
        ests.append(("PE", per[1], per[0], final_val))
        if DEBUG:
            print("Hist avg PE: {}  Fwd EPS: {}  DV Est {} {}: {}".format('%.3f' % (data['ols']['pe_5yr_avg_hist'][period]),
                '%.3f' % (data['ols']['eps'][per]), per[1], per[0], final_val))

    # P/S
    # Sales per share
    data['ols']['sps'] = data['ols']['revenue'] / data['ols']['weight_avg_shares'] 
    data['ols']['ps_avg_hist'] = data['ols']['avg_52wk'] / data['ols']['sps']
    data['ols']['ps_curr_hist'] = curr_px / data['ols']['sps']
    data['ols']['ps_fwd'] = ((data['ols']['date_px'] * data['is']['weight_avg_shares'])
                             / data['is']['revenue'].shift(1))
    data['ols']['ps_5yr_avg_hist'] = data['ols']['ps_avg_hist'].rolling(center=False, window=5).mean()
    
    for per in [next_per, pers_2]:
        final_val = '%.3f' % (data['ols']['ps_5yr_avg_hist'][period] * (data['ols']['sps'][per]))
        ests.append(("PS", per[1], per[0], final_val))
        if DEBUG:
            print("Hist avg PS: {}  Fwd Rev/share: {}  DV Est {} {}: {}".format('%.3f' % (data['ols']['ps_5yr_avg_hist'][period]),
                '%.3f' % (data['ols']['sps'][per]), per[1], per[0], final_val))

    pdb.set_trace()
    # P/B
    data['ols']['pb_avg_hist'] = ((data['ols']['avg_52wk']
                                   * data['is']['weight_avg_shares']) / data['is']['revenue'])
    data['ols']['pb_curr_hist'] = ((curr_px * data['is']['weight_avg_shares'])
                                   / data['is']['revenue'])
    data['ols']['pb_fwd'] = ((data['ols']['date_px'] * data['is']['weight_avg_shares'])
                             / data['is']['revenue'].shift(1))
    data['ols']['pb_5yr_avg_hist'] = data['ols']['ps_avg_hist'].rolling(center=False, window=5).mean()
    data['ols']['bvps'] = data['ols']['revenue'] / data['ols']['weight_avg_shares']  # Sales per share
    
    for per in [next_per, pers_2]:
        final_val = '%.3f' % (data['ols']['ps_5yr_avg_hist'][period] * (data['ols']['sps'][per]))
        ests.append(("PS", per[1], per[0], final_val))
        if DEBUG:
            print("Hist avg PS: {}  Fwd Rev/share: {}  DV Est {} {}: {}".format('%.3f' % (data['ols']['ps_5yr_avg_hist'][period]),
                '%.3f' % (data['ols']['sps'][per]), per[1], per[0], final_val))
                
    data['PB'] = (data['52WeekAvg'] * data['shares']) / data['totalEquity']
    data['PB_curr'] = (data['currentPrice'] * data['shares']) / data['totalEquity']
    data['PB_fwd'] = (data['currentPrice'] * data['shares']) / data['totalEquity'].shift(1)
    data['PB_5yr_avg_hist'] = data['PB'].rolling(center=False, window=5).mean()
    for p in [next_per, pers_2]:
        final_val = '%.3f'%(data['PB_5yr_avg_hist'][period] * data['totalEquity'][p] / data['shares'][period])
        ests.append(("PB", p[1], p[0], final_val))
        if DEBUG:
            print("Hist avg PB: {}  Fwd equity/share: {}  DV Est {} {}: {}".format('%.3f'%(data['PB_5yr_avg_hist'][period]),
                '%.3f'%(data['totalEquity'][p] / data['shares'][period]), p[1], p[0], final_val))

    # P/CF
    data['PCF'] = (data['52WeekAvg'] * data['shares']) / data['operCF']
    data['PCF_curr'] = (data['currentPrice'] * data['shares']) / data['operCF']
    data['PCF_fwd'] = (data['currentPrice'] * data['shares']) / data['operCF'].shift(1)
    data['PCF_5yr_avg_hist'] = data['PCF'].rolling(center=False, window=5).mean()
    for p in [next_per, pers_2]:
        final_val = '%.3f'%(data['PCF_5yr_avg_hist'][period] * data['operCF'][p] / data['shares'][period])
        ests.append(("PCF", p[1], p[0], final_val))
        if DEBUG:
            print("Hist avg PCF: {}  Fwd CF/share: {}  DV Est {} {}: {}".format('%.3f'%(data['PCF_5yr_avg_hist'][period]),
                '%.3f'%(data['operCF'][p] / data['shares'][period]), p[1], p[0], final_val))

    # P/FCF
    data['PFCF'] = (data['52WeekAvg'] * data['shares']) / data['FCF']
    data['PFCF_curr'] = (data['currentPrice'] * data['shares']) / data['FCF']
    data['PFCF_fwd'] = (data['currentPrice'] * data['shares']) / data['FCF'].shift(1)
    data['PFCF_5yr_avg_hist'] = data['PFCF'].rolling(center=False, window=5).mean()
    for p in [next_per, pers_2]:
        final_val = '%.3f'%(data['PFCF_5yr_avg_hist'][period] * data['FCF'][p] / data['shares'][period])
        ests.append(("PFCF", p[1], p[0], final_val))
        if DEBUG:
            print("Hist avg PFCF: {}  Fwd FCF/share: {}  DV Est {} {}: {}".format('%.3f'%(data['PFCF_5yr_avg_hist'][period]),
                '%.3f'%(data['FCF'][p] / data['shares'][period]), p[1], p[0], final_val))

    # Relative P/E
    # NEED THE EARNIGNS OF THE SNP500
    # data['PE_rel'] = (data['52WeekAvg'] * data['shares']) / data['PE_of_SnP']
    # data['PE_rel_curr'] = (data['currentPrice'] * data['shares']) / data['PE_of_SnP']
    # data['PE_rel_fwd'] = (data['currentPrice'] * data['shares']) / data['PE_of_SnP'].shift(1)
    # data['PE_rel__5yr_avg'] = data['PE_rel'].rolling(center=False, window=5).mean()
    # for p in [next_per, pers_2]:
        # print("Hist avg PS: {}  Fwd Rev/share: {}  DV Est {} {}: {}".format(data['PE_rel__5yr_avg'][period],
        #     data['PE_of_SnP'][p] / data['shares'][period], period[1], period[0]
        #     data['PE_rel__5yr_avg'][period] * data['revenue'][p] / data['shares'][period]))

    # PEG
    data['PEGY'] = data['PE_avg_hist'] / ((data['netIncome'].pct_change() + data['divYield']) * 100)
    data['PEGY_5yr_avg'] = data['PEGY'].rolling(center=False, window=5).mean()
    return data, ests


def ratios_and_valuation(data, eod_px):
    """
    Add some necessary columns
    """
    # get price info
    data = match_px(data, eod_px)

    # Balance Sheet Columns
    data['bs']['div_per_share'] = (data['cf']['divs_paid']
                                   / data['is']['weight_avg_shares'])

    # Income Statement columns
    # Net Operatin Profit after tax
    data['is']['nopat'] = (data['is']['oper_inc']
                           * (1 - data['fr']['eff_tax_rate']))
    data['is']['rtc'] = data['is']['nopat'] / data['bs']['total_assets']

    # Financial Ratios
    data['fr']['trade_work_cap'] = (data['bs']['receivables']
                                    + data['bs']['inv']
                                    - data['bs']['accounts_payable'])
    data['fr']['ebitda_margin'] = data['is']['ebitda'] / data['is']['revenue']
    data['fr']['ret_earn_ratio'] = (1 - data['fr']['div_payout_ratio'])
    data['fr']['const_growth_rate'] = (data['fr']['roe'] 
                                       * data['fr']['ret_earn_ratio'])
    data['fr']['oper_lev'] = (data['is']['oper_inc'].pct_change()
                              / data['is']['revenue'].pct_change())
    data['fr']['roe_dupont'] = ((data['is']['net_inc'] / data['is']['revenue'])
                                * (data['is']['revenue'] / data['bs']['total_assets'])
                                * (data['bs']['total_assets'] / data['bs']['total_equity']))

    # Cash Flow Statement Columns
    data['cf']['equity_turnover'] = (data['is']['revenue']
                                    / data['bs']['total_equity'])
    data['cf']['cash_conv_cycle'] = (data['fr']['days_of_inv_on_hand']
                                     + data['fr']['days_sales_outstanding']
                                     - data['fr']['days_payables_outstanding'])
    data['cf']['fcf_min_wc'] = data['cf']['fcf'] - data['cf']['chg_working_cap']
    data['cf']['fcf_min_twc'] = (data['cf']['fcf']
                                 - data['fr']['trade_work_cap'].diff())
    return data


def model_est_ols(years, data, avg_cols=None, use_last=None):
    """
    Create a model based on ordinary least squares regression
    """
    hist = pd.DataFrame()
    data_ols = pd.DataFrame()
    # some cleanup
    for sheet in ['is', 'bs', 'cf', 'fr']:
        data[sheet] = data[sheet].reset_index()[data[sheet].reset_index().year != 'TTM'].set_index(IDX)

    # next qurater est is equal to revenue * average est margin
    # over the last year
    for _ in range(years):
        if hist.empty:
            n_idx = list(data['is'].iloc[-1].name)
        else:
            n_idx = list(hist.iloc[-1].name)
        n_idx = getNextYear(n_idx)
        n_hist_dict = {k: v for k, v in zip(IDX, n_idx)}

        #########
        # Use OLS to get projected values
        #########
        for cat in OLS_COLS:
            # for columns that are all 0 for a particular security
            skip = False
            # Need this for columns that are too eradic for OLS
            if avg_cols and cat in avg_cols:
                n_hist_dict[cat] = data[cat].mean()
                continue
            # Need this for columns where we just use most recent value
            if use_last and cat in use_last:
                n_hist_dict[cat] = data[cat].values[-1]
                continue
            x_val = data['is'].reset_index()[['year', 'month']]
            for sheet in ['is', 'bs', 'cf', 'fr']:
                try:
                    val = data[sheet][cat]
                    data_ols[cat] = val
                    break
                except KeyError as exc:
                    if sheet == 'fr':
                        n_hist_dict[cat] = 0
                        data_ols[cat] = 0
                        skip = True
                    else:
                        continue
            # column is 0 for this security
            if skip:
                continue
            slope, yint = ols_calc(x_val, val)
            start = dt.datetime(int(x_val.values[0][0]),
                                int(x_val.values[0][1]), 1).date()
            new_x = get_year_deltas([start, dt.datetime(int(n_idx[0]),
                                                        int(n_idx[2][:-1]),
                                                        1).date()])[-1]
            # Need this to convert terminology for quarterly, also need to divide by four
            n_hist_dict[cat] = (yint + new_x * slope)

        n_hist_dict['ebt'] = (n_hist_dict['oper_inc'] 
                              + n_hist_dict['net_int_inc'])
        # assume average tax rate over last 5 years
        n_hist_dict['taxes'] = (data['fr']['eff_tax_rate'].mean()
                                * n_hist_dict['ebt'])
        n_hist_dict['net_inc'] = n_hist_dict['ebt'] - n_hist_dict['taxes']
        n_hist_dict['eps'] = (n_hist_dict['net_inc']
                              / n_hist_dict['weight_avg_shares'])
        t_df = pd.DataFrame(n_hist_dict, index=[0]).set_index(IDX)
        hist = hist.append(t_df)
        
    hist = pd.concat([data_ols, hist])
    return hist


def model_est(cum, margin):
    """
    Model based on quarterly estimats
    """
    # some cleanup
    margin = margin.reset_index()[margin.reset_index().date != 'TTM'].set_index(IDX)
    cum = cum.reset_index()[cum.reset_index().month != ''].set_index(IDX)

    # next qurater est is equal to revenue * average est margin over the last year
    for _ in range(4):
        n_idx = list(cum.iloc[-1].name)
        n_idx = getNextQuarter(n_idx)
        n_data = n_idx + list(margin[-5:-1].mean())
        t_df = pd.DataFrame(dict((key, value) for (key, value) in zip(IDX+list(margin.columns), n_data)), columns=IDX+list(margin.columns), index=[0]).set_index(IDX)
        margin = margin.append(t_df)

        n_cum_dict = {k: v for k, v in zip(IDX, n_idx)}
        n_cum_dict['revenue'] = cum[-5:-1]['revenue'].mean()

        #########
        # Use mean of previous few years to get there
        #########
        for col in ['cogs', 'rd', 'sga', 'grossProfit', 'mna', 'otherExp']:
            n_cum_dict[col] = margin.loc[tuple(n_idx)][col] * n_cum_dict['revenue']
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

        t_df = pd.DataFrame(n_cum_dict, index=[0]).set_index(IDX)
        cum = cum.append(t_df)

    # clean up df
    cum = cum.fillna(0)
    empty_cols = [c for c in list(cum.columns) if all(v == 0 for v in cum[c])]
    cum = cum[list(set(cum.columns) - set(empty_cols))]
    return cum


def margin_df(marg_df):
    """
    calculates margin for these columns
    """
    data = marg_df.apply(lambda x: x / x['revenue'], axis=1)
    # tax rate presented as percentage of pre tax income
    data['taxes'] = marg_df['prov_inc_tax'] / marg_df['ebt']
    return data


def ols_calc(xvals, yvals, n_idx=None):
    """
    ordinary least squares calculation
    """
    xvals = [dt.datetime(int(x[0]), int(float(str(x[1]).replace("E", ""))),
                      1).date() for x in xvals.values]
    start = xvals[0]
    xvals = get_year_deltas(xvals)
    A_mat = np.vstack([xvals, np.ones(len(xvals))]).T
    slope, yint = np.linalg.lstsq(A_mat, yvals.values)[0]
    return (slope, yint)


def get_price_data(ticks, comps, method='db'):
    """
    Grab data from API, File, or DB
    """
    pxs = pd.DataFrame()
    # Cant find an API I can trust for EOD stock data
    for ind_t in ticks + comps:
        if method == 'api':
            start = dt.date(2000, 1, 1).strftime("%Y-%m-%d")
            end = dt.datetime.today().date().strftime("%Y-%m-%d")
            url = "https://www.quandl.com/api/v1/datasets/WIKI/{0}.csv?column=4&sort_order=asc&trim_start={1}&trim_end={2}".format(ind_t, start, end)
        elif method == 'file':
            qrd = pd.read_csv("/home/ubuntu/workspace/python_for_finance/research/data_grab/{}.csv".format(ind_t))
            qrd = qrd.rename(columns={'close': ind_t}).set_index(['date'])
            if pxs.empty:
                pxs = qrd
            else:
                pxs = pd.merge(pxs, qrd, how='left',
                               left_index=True, right_index=True)
        else:
            pxs = get_ticker_info(ticks + comps, 'eod_px', ['tick', 'date'])
            break
    return pxs


def analyze_ests(data, period, hist_px, years_fwd=2):
    val_models = ['Hist Comps', 'DFCF', 'PDV']
    val_weights = {
        'Hist Comps': 0.35,
        'DFCF': 0.5,
        'PDV': 0.15
    }
    for k, v in data.items():
        per = tuple([period[0], k, v[0].index.values[0][2]])
        print("Tick: {}   Date: {} {}".format(k, per[0], per[2]))
        print("Current Price: {}".format(hist_px[k].dropna().values[-1]))
        try:
            v[0]['beta'][per]
        except:
            # Company may havent reported yet in current year
            per = tuple([str(int(period[0])-1), k, v[0].index.values[0][2]])

        for y in range(1, years_fwd+1):
            year = str(int(per[0])+y)
            year_est = {}
            for mod in val_models:
                mod_est = []
                for val in valuations[mod]:
                    try:
                        e = [float(est[3]) for est in v[1] if est[0] == val and
                             est[1] == k and est[2] == year][0]
                    except:
                        # Might not have this model for this year
                        continue
                    mod_est.append(e)
                    print("Model: {}  tick: {}  year: {}  EST: {}".format(val, k, year, e))
                if len(mod_est) == 0:
                    continue
                year_est[mod] = sum(mod_est)/len(mod_est)
                print("Models AVG: {}  tick: {}  year: {}  EST: {}".format(mod, k, year, '%.4f' % year_est[mod]))
                prem_disc = (year_est[mod] / float(hist_px[k].dropna().values[-1])) - 1
                # Divide by beta
                risk_adj = ((year_est[mod] / float(hist_px[k].dropna().values[-1])) - 1) / v[0]['beta'][per]
                print("Prem/Disc to Current PX: {}  Risk Adj Prem/Disc: {}".format('%.4f' % prem_disc, '%.4f' % risk_adj))
            # Assume 50% for DFCF, 35% for Hist comparables, 15% for peer derived
            year_avg_est = 0
            if len(list(year_est.keys())) < 3:
                # dont have values for all models
                continue
            for key, estimate in year_est.items():
                year_avg_est += estimate * val_weights[key]
            print("Current Price: {}".format(hist_px[k].dropna().values[-1]))
            print("Weighted AVG Estimate   tick: {}  year: {}  EST: {}".format(k, year, '%.4f'%year_avg_est))
            prem_disc = (year_avg_est
                         / float(hist_px[k].dropna().values[-1])) - 1
            # Divide by beta
            risk_adj = ((year_avg_est / float(hist_px[k].dropna().values[-1])) - 1) / v[0]['beta'][per]
            print("Prem/Disc to Current PX: {}  Risk Adj Prem/Disc: {}".format('%.4f'%prem_disc, '%.4f'%risk_adj))


def valuation_model(ticks, mode='db'):
    """
    Main method for valuation model
    """
    full_data = {}

    ind = 'XLK'
    mkt = 'SPY'
    other = [mkt, ind]
    # Get Historical Price data
    hist_px = get_price_data(ticks, other, mode)

    for ind_t in ticks:
        data = {}
        data['is'] = get_ticker_info([ind_t], 'inc_statement',
                                     ['year', 'month', 'tick'])
        data['bs'] = get_ticker_info([ind_t], 'bal_sheet',
                                     ['year', 'month', 'tick'])
        data['cf'] = get_ticker_info([ind_t], 'cf_statement',
                                     ['year', 'month', 'tick'])
        data['fr'] = get_ticker_info([ind_t], 'fin_ratios',
                                     ['year', 'month', 'tick'])
        # join on whatever data you need
        data['is']['gross_profit'] = data['is']['revenue'] - data['is']['cogs']

        data = removeEmptyCols(data)
        # This is only if we have quarterly data
        # data_cum = dataCumColumns(data)
        data_chg = period_chg(data)
        data_margin = margin_df(data['is'])[['gross_profit', 'cogs', 'rnd',
                                             'sga', 'restruct_mna',
                                             'prov_inc_tax', 'other_oper_exp']]

        # project out data based on historicals using OLS regression
        data['ols'] = model_est_ols(10, data)

        # Add some columns and adjustemnts and calculate ratios for Use later
        data = ratios_and_valuation(data, hist_px)

        period = [i for i in data['is'].index.values if "E" not in i[2]][-1]
        # Get Historical Ratios for valuation
        hist_rats, ests = historical_ratios(data, period, hist_px)
        # Discounted Free Cash Flow Valuation
        pdb.set_trace()
        dfcf, ests = discount_fcf(hist_rats, period, ests, hist_px)
        full_data[t] = [dfcf, ests]

    # calculate performance metrics based on price
    px_df = price_perf_anal(period, mkt, ind, hist_px)

    # Comaprisons
    comp_anal = comparison_anal(full_data, period)

    # Peer Derived Value
    full_data, pdv = peer_derived_value(full_data, comp_anal, period, hist_px)

    # Analysis of all valuations
    analyze_ests(full_data, period, hist_px)


if __name__ == '__main__':
    # income_state_model(['MSFT'], 'api')
    # valuation_model(['MSFT'])
    valuation_model(['MSFT', 'AAPL', 'CSCO', 'INTC', 'ORCL'])
    