"""
Script to perform equity research and analysis as described in
Equity Valuation for Analysts and Investors by James Kelleher
"""
import sys
import pdb
import warnings
import datetime as dt
import numpy as np
import pandas as pd
import quandl
quandl.ApiConfig.api_key = 'J4d6zKiPjebay-zW7T8X'

sys.path.append("/home/ec2-user/environment/python_for_finance/")
from res_utils import get_ticker_info, remove_empty_cols, setup_comp_cols, \
                      get_next_year, OLS_COLS, VALUATIONS, get_next_quarter, \
                      get_beta, setup_pdv_cols, match_px, replace_needed_cols
from dx.frame import get_year_deltas

# NOTES:
# simulate api call --> http://financials.morningstar.com/ajax/exportKR2CSV.html?t=BA

IDX = ['year', 'tick', 'month']
DEBUG = False
STOCK_DEBUG = True
STEP_THRU = True
warnings.filterwarnings("ignore")

PDV_MAP = {
    "ps_ratio": "PS",
    "pe_avg_hist": "PE",
    "pb_ratio": "PB",
    "pfcf_ratio": "PCF"
}


def peer_derived_value(data, period):
    """
    Get the value of the stock as compared to its peers
    """
    # get group values first
    group_vals = {}
    years_fwd = 2
    vals = ['ps_ratio', 'pe_avg_hist', 'pb_ratio', 'pfcf_ratio']
    ticks = list(data.keys())

    if STEP_THRU:
        pdb.set_trace()
        pass

    # get group market cap
    group_mkt_cap = 0
    for key, data_df in data.items():
        per = tuple([period[0], key, data[key][0]['ols'].index.values[0][2]])
        try:
            group_mkt_cap += (data_df[0]['fr']['market_cap']).reset_index().set_index('year')['market_cap']
        except KeyError:
            # May not have reported yet for this year, if this also fails,
            # raise exception as may have bigger problem
            # pdb.set_trace()
            per = tuple([str(int(period[0])), key, period[2]+"E"])

        # Need to project out vals
        for ind_val in vals + ['market_cap']:
            if ind_val in ['pe_avg_hist']:
                sheet = 'ols'
            else:
                sheet = 'fr'
            xvals = data_df[0][sheet][ind_val].dropna().reset_index()[['year', 'month']]
            month = xvals['month'].values[0]
            slope, yint = ols_calc(xvals,
                                   data_df[0][sheet][ind_val].dropna().astype('float'))
            for fwd in range(1, years_fwd + 1):
                start = dt.datetime(int(xvals.values[0][0]),
                                    int(xvals.values[0][1]), 1).date()
                per = tuple([str(int(period[0]) + fwd), key, month+"E"])
                new_x = get_year_deltas([start, dt.datetime(int(per[0]),
                                                            int(per[2][:-1]), 1).date()])[-1]
                data_df[0][sheet].at[per, ind_val] = (yint + new_x * slope)

    # ols for group market cap
    xvals = group_mkt_cap.reset_index()
    xvals['month'] = '06'
    month = '06'
    xvals = xvals[['year', 'month']]
    slope, yint = ols_calc(xvals,
                           group_mkt_cap.astype('float'))
    for fwd in range(1, years_fwd + 1):
        start = dt.datetime(int(xvals.values[0][0]),
                            int(xvals.values[0][1]), 1).date()
        per = tuple([str(int(period[0]) + fwd), month+"E"])
        new_x = get_year_deltas([start, dt.datetime(int(per[0]),
                                                    int(month), 1).date()])[-1]
        group_mkt_cap[per[0]] = (yint + new_x * slope)

    for ind_val in vals:
        if ind_val in ['pe_avg_hist']:
            sheet = 'ols'
        else:
            sheet = 'fr'
        group_vals[ind_val] = 0
        group_vals[ind_val + "_w_avg"] = 0
        for yrf in range(1, years_fwd + 1):
            group_vals[ind_val + "_" + str(yrf) + "fwd"] = 0
            group_vals[ind_val + "_" + str(yrf) + "fwd_w_avg"] = 0

        for tick in ticks:
            per = tuple([period[0], tick,
                         data[tick][0][sheet].index.values[0][2]])
            fwd_pers = [tuple([str(int(period[0]) + yf), tick,
                               data[tick][0][sheet].index.values[0][2] + "E"])
                        for yf in range(1, years_fwd + 1)]
            try:
                # 5yr avgs, simple and weighted
                group_vals[ind_val] += data[tick][0][sheet][ind_val].dropna().rolling(center=False, window=5).mean()[per] / len(ticks)
                group_vals[ind_val + "_w_avg"] += (data[tick][0][sheet][ind_val].dropna().rolling(center=False, window=5).mean()[per]
                                                   * (data[tick][0]['fr']['market_cap'][per]
                                                      / group_mkt_cap[per[0]]))
            except KeyError:
                # May not have reported yet for this year, if this also fails,
                # raise exception as may have bigger problem
                # pdb.set_trace()
                per = tuple([str(int(period[0])-1), tick,
                             data[tick][0].index.values[0][2]])

            for fwd in fwd_pers:
                year_diff = int(fwd[0]) - int(per[0])
                # fwd avgs, simple and weighted
                group_vals[ind_val + "_" + str(year_diff) + "fwd"] += data[tick][0][sheet][ind_val].dropna().rolling(center=False, window=years_fwd).mean()[fwd] / len(ticks)
                group_vals[ind_val + "_" + str(year_diff) + "fwd_w_avg"] += (data[tick][0][sheet][ind_val].dropna().rolling(center=False, window=years_fwd).mean()[fwd]
                                                                             * (data[tick][0]['fr']['market_cap'][fwd]
                                                                                / group_mkt_cap[fwd[0]]))
        if DEBUG:
            print("{} 5Y simple avg: {}".format(ind_val, '%.3f' % group_vals[ind_val]))
            print("{} 5Y weighted avg: {}".format(ind_val, '%.3f' % group_vals[ind_val + "_w_avg"]))
            for yrf in range(1, years_fwd + 1):
                print("{} {}Y fwd avg: {}"
                      "".format(ind_val, str(yrf),
                                '%.3f' % group_vals[ind_val + "_" + str(yrf) + "fwd"]))
                print("{} {}Y fwd weighted avg: {}"
                      "".format(ind_val, str(yrf),
                                '%.3f' % group_vals[ind_val + "_" + str(yrf) + "fwd_w_avg"]))

    comp_df = pd.DataFrame()
    for key, data_df in data.items():
        per = tuple([period[0], key, data_df[0]['ols'].index.values[0][2]])
        for ratio in vals:
            if ratio in ['pe_avg_hist']:
                sheet = 'ols'
            else:
                sheet = 'fr'
            if comp_df.empty:
                comp_df = pd.DataFrame(columns=setup_pdv_cols(per, years_fwd))
            row = [key, ratio]
            try:
                # 5y average
                row.append(data_df[0][sheet][ratio].dropna().rolling(center=False, window=5).mean()[per])
            except KeyError:
                # May not have reported yet for this year, if this also fails,
                # raise exception as may have bigger problem
                # pdb.set_trace()
                per = tuple([str(int(period[0])-1),
                             key, data[key][0].index.values[0][2]])

            # 5y avg vs weighted avg
            row.append(row[-1] / group_vals[ratio + "_w_avg"])

            # get fwd years
            fwd_pers = [tuple([str(int(period[0]) + yf), key,
                               data_df[0]['ols'].index.values[0][2] + "E"])
                        for yf in range(1, years_fwd+1)]
            for fwd in fwd_pers:
                year_diff = int(fwd[0]) - int(per[0])
                # fwd multiple
                row.append(data_df[0][sheet][ratio].dropna().rolling(center=False, window=year_diff).mean()[fwd])
                # fwd mult vs fwd group average
                row.append(row[-1] / group_vals[ratio + "_" + str(year_diff) + "fwd_w_avg"])
                # premium / discount: (5yr avg / group wgt avg)
                #                      / (fwd mult vs fwd group wgt ratio)
                # aka relative fwd mult compared to current mult
                row.append(row[3] / row[-1])
                # prem_discount * current price for Peer derived value
                row.append(data_df[0]['ols'].date_px[per] * row[-1])
                data[key][1].append(tuple(["pdv_" + ratio, per[1],
                                           str(int(per[0]) + year_diff), '%.3f' % row[-1]]))
            comp_df.loc[len(comp_df)] = row
    return data, comp_df.set_index(['ticker', 'cat'])


def comparison_anal(data, period):
    """
    doing comparison analysis on security
    """
    comp_df = pd.DataFrame()
    years_back = 5
    years_fwd = 2
    cols = ['net_inc', 'revenue', 'gross_prof_marg', 'pretax_prof_marg',
            'net_prof_marg', 'pe_avg_hist']
    cagr = ['net_inc', 'revenue']

    if STEP_THRU:
        pdb.set_trace()
        pass

    for tick, dfs in data.items():
        indices = [d for d in list(dfs[0]['ols'].index.values)
                   if int(d[0]) > int(period[0]) - years_back
                   and int(d[0]) <= int(period[0]) + years_fwd]
        ind_df = dfs[0]['ols'].ix[indices]
        if comp_df.empty:
            comp_df = pd.DataFrame(columns=setup_comp_cols(indices))
        for cat in cols:
            if cat in cagr:
                # cagr = compound annual growth rate
                avg = (ind_df[cat][years_back-1] / ind_df[cat][0])**(1/5) - 1
                row = [tick, cat] + list(ind_df[cat].values)
                row.insert(2 + years_back, avg)
                comp_df.loc[len(comp_df)] = row

                # standard avg of growth rate
                avg = ind_df[cat].pct_change()[1:years_back].mean()
                row = [tick, cat + "_g"] + list(ind_df[cat].pct_change().values)
                row.insert(2 + years_back, avg)
                comp_df.loc[len(comp_df)] = row
            else:
                # standard avg
                avg = ind_df[cat][:years_back].mean()
                row = [tick, cat] + list(ind_df[cat].values)
                row.insert(2+years_back, avg)
                comp_df.loc[len(comp_df)] = row
    return comp_df.set_index(['ticker', 'cat'])


def price_perf_anal(period, mkt, ind, hist_px):
    """
    Compare performance of stocks to their indices
    """
    if STEP_THRU:
        pdb.set_trace()
        pass

    px_df = pd.DataFrame()
    mkt_px = hist_px.loc[mkt]
    ind_px = hist_px.loc[ind]
    for ind_t in list(set(hist_px.index.get_level_values(0).unique()) - set([mkt, ind])):
        t_px = hist_px.loc[ind_t]
        potential_yrs = list(set([ind_dt.year for ind_dt in list(t_px.index)
                                  if ind_dt.year > int(period[0])]))
        for yrs in [int(period[0])] + potential_yrs:
            t_df = pd.DataFrame([ind_t], columns=['tick'])
            t_df['year'] = yrs
            year_px = (t_px[(t_px.index >= dt.datetime(yrs, 1, 1))
                            & (t_px.index <= dt.datetime(yrs, 12, 31))])
            t_df['cur_px'] = year_px.values[-1][0]
            t_df['y_px'] = year_px.values[0][0]
            t_df['ytd_chg'] = (t_df['cur_px'] / t_df['y_px']) - 1
            t_df['ytd_high'] = max(year_px.values)
            t_df['ytd_low'] = min(year_px.values)
            year_mkt_px = (mkt_px[(mkt_px.index >= dt.datetime(yrs, 1, 1))
                                  & (mkt_px.index <= dt.datetime(yrs, 12, 31))])
            year_ind_px = (ind_px[(ind_px.index >= dt.datetime(yrs, 1, 1))
                                  & (ind_px.index <= dt.datetime(yrs, 12, 31))])
            t_df['mkt_ytd_chg'] = (year_mkt_px.values[-1][0]
                                   / year_mkt_px.values[0][0] - 1)
            t_df['mkt_rel_perf'] = t_df['ytd_chg'] - t_df['mkt_ytd_chg']
            t_df['ind_ytd_chg'] = (year_ind_px.values[-1][0]
                                   / year_ind_px.values[0][0] - 1)
            t_df['ind_rel_perf'] = t_df['ytd_chg'] - t_df['ind_ytd_chg']
            px_df = px_df.append(t_df)
    return px_df.set_index(['tick', 'year'])


def discount_fcf(data, period, ests, stock):
    """
    Calculate the value of the security based on
    discounted free cash flow models
    """
    # should be pulled off debt issued by company, hard coded for now
    cost_debt = 0.07
    # rate on 10yr tsy
    r_free = 0.028
    # generally accepted market risk premium above risk free rate
    market_risk_prem = 0.05

    # Take the periods beta, calc'd in res_utils
    # mean of last 3 years beta
    beta = data['ols']['beta'].dropna()[-3:].mean()
    
    model = 'DIV_CAP'
    if model == 'CAPM':
        # CAPM
        cost_equity = r_free + beta * market_risk_prem
    elif model == 'DIV_CAP':
        # Dividend Capitalization Model
        div_g = 0.15
        cost_equity = (data['ols']['div_per_share'][period]
                       / data['ols']['date_px'][period]) + div_g
    
    
    
    mv_eq = data['fr']['market_cap'][period]
    # mv_debt = HARD TO GET
    bv_debt = (data['bs']['short_term_debt'][period]
               + data['bs']['long_term_debt'][period])
    total_capital = bv_debt + mv_eq
    eq_v_cap = mv_eq / total_capital
    debt_v_cap = bv_debt / total_capital
    # average of the periods we have
    tax_rate = data['fr']['eff_tax_rate'].mean()
    wacc = ((cost_equity * eq_v_cap) + (cost_debt * debt_v_cap)
            * (1 - tax_rate / 100))
    if DEBUG or (STOCK_DEBUG and stock == period[1]):
        print("WACC: " + str(wacc))

    if STEP_THRU and stock == period[1]:
        pdb.set_trace()
        pass

    # todo: ENTER analysts projected EPS growth here
    eps_g_proj = 0.12
    # average of calc'd growth and analyst projection
    data['ols']['proj_calc_g'] = (data['fr']['const_growth_rate'] + eps_g_proj) / 2
    # avg of constant growth calc
    data['ols']['1st_5yr_lt_g'] = data['fr']['const_growth_rate'].mean()
    # slightly lower than 1st growth calc, usually 2 to 4%
    data['ols']['2nd_5yr_lt_g'] = data['ols']['1st_5yr_lt_g'] - 0.02
    # long term terminal growth rate
    # typically some average of gdp and the industry standard
    term_growth = 0.05

    # 2 Stage DFCF
    years_to_terminal = [1, 2]
    for ytt in years_to_terminal:
        fcf_pershare = (data['cf']['fcf_min_twc'][period]
                        / data['is']['weight_avg_shares'][period])
        indices = [d for d in list(data['ols'].index.values)
                   if int(d[0]) > int(period[0]) and int(d[0]) <= int(period[0]) + ytt]
        # fcf geometric growth
        fcfs = [fcf_pershare * (1 + data['ols']['1st_5yr_lt_g'][period])
                ** (int(x[0]) - int(period[0])) for x in indices]
        disc_fcfs = [fcfs[x] / (1 + cost_equity) **
                     (int(indices[x][0]) - int(period[0])) for x in range(0, len(indices))]
        sum_of_disc_cf = sum(disc_fcfs)
        term_val = ((data['ols']['fcf_min_twc'][indices[-1]]
                     / data['is']['weight_avg_shares'][period]) / (cost_equity - term_growth))
        final_val = term_val + sum_of_disc_cf
        ests.append(("2stage", indices[-1][1], indices[-1][0],
                     '%.3f' % (final_val)))
        if DEBUG or (STOCK_DEBUG and stock == period[1]):
            print("2 Stage Val Est {} {}: {}".format(indices[-1][1],
                                                     indices[-1][0], '%.3f'%(final_val)))

    # 3 Stage DFCF
    years_to_terminal_3 = [[0, 1], [1, 1]]
    for ytt in years_to_terminal_3:
        # 1st growth phase
        fcf_pershare = (data['ols']['fcf_min_twc'][period]
                        / data['is']['weight_avg_shares'][period])
        indices = [d for d in list(data['ols'].index.values) if int(d[0]) > int(period[0])
                   and int(d[0]) <= int(period[0]) + ytt[0]]
        fcfs = [fcf_pershare * (1 + data['ols']['1st_5yr_lt_g'][period])
                ** (int(x[0]) - int(period[0])) for x in indices]
        disc_fcfs_1 = [fcfs[x] / (1 + cost_equity) **
                       (int(indices[x][0]) - int(period[0])) for x in range(0, len(indices))]
        # second growth phase
        # need to make sure hav the right indices after first growth period is over
        indices = [d for d in list(data['ols'].index.values)
                   if int(d[0]) > int(period[0]) + ytt[0]
                   and int(d[0]) <= int(period[0]) + ytt[0] + ytt[1]]
        fcfs = [fcf_pershare * (1 + data['ols']['2nd_5yr_lt_g'][period])
                ** (int(x[0]) - int(period[0])) for x in indices]
        disc_fcfs_2 = [fcfs[x] / (1 + cost_equity) **
                       (int(indices[x][0]) - int(period[0])) for x in range(0, len(indices))]
        sum_of_disc_cf = sum(disc_fcfs_1) + sum(disc_fcfs_2)
        term_val = ((data['ols']['fcf_min_twc'][indices[-1]]
                     / data['is']['weight_avg_shares'][period]) / (cost_equity - term_growth))
        final_val = term_val + sum_of_disc_cf
        ests.append(("3stage", indices[-1][1], indices[-1][0],
                     '%.3f' % (final_val)))
        if DEBUG or (STOCK_DEBUG and stock == period[1]):
            print("3 Stage Val Est {} {}: {}".format(indices[-1][1],
                                                     indices[-1][0], '%.3f'%(final_val)))

    # Component DFCF
    years_to_terminal = [1, 2]
    for ytt in years_to_terminal:
        # use the OLS growth calcs for FCFs instead of growth forecasts
        fcfs = (data['ols']['fcf_min_twc'] / data['ols']['weight_avg_shares'])
        indices = [d for d in list(data['ols'].index.values) if int(d[0]) > int(period[0])
                   and int(d[0]) <= int(period[0]) + ytt]
        disc_fcfs = [fcfs[indices[x]] / (1 + cost_equity) **
                     (int(indices[x][0]) - int(period[0])) for x in range(0, len(indices))]
        sum_of_disc_cf = sum(disc_fcfs)
        term_val = (data['ols']['fcf_min_twc'][indices[-1]]
                    / data['is']['weight_avg_shares'][period]) / (cost_equity - term_growth)
        final_val = term_val + sum_of_disc_cf
        ests.append(("Component Anal", indices[-1][1], indices[-1][0],
                     '%.3f' % (final_val)))
        if DEBUG or (STOCK_DEBUG and stock == period[1]):
            print("Component Val {} {}: {}".format(indices[-1][1],
                                                   indices[-1][0], '%.3f'%(final_val)))
    return data, ests


def historical_ratios(data, period, hist_px, stock):
    """
    Calculate historical ratios for valuation
    """
    ests = []
    next_per = tuple(get_next_year(period))
    pers_2 = tuple(get_next_year(next_per))

    if STEP_THRU and stock == period[1]:
        pdb.set_trace()
        pass

    # fill current price with latest measurement
    curr_px = hist_px.loc[period[1]].iloc[-1]['px']

    # PE Ratios
    data['ols']['eps'] = data['ols']['net_inc'] / data['ols']['weight_avg_shares']
    data['ols']['pe_low_hist'] = data['ols']['lo_52wk'] / data['ols']['eps']
    data['ols']['pe_low_hist'] = data['ols']['hi_52wk'] / data['ols']['eps']
    data['ols']['pe_avg_hist'] = data['ols']['avg_52wk'] / data['ols']['eps']
    data['ols']['pe_curr_hist'] = curr_px / data['ols']['eps']
    data['ols']['pe_fwd'] = ((data['ols']['date_px']
                              * data['is']['weight_avg_shares'])
                             / data['is']['net_inc'].shift(1))
    data['ols']['pe_5yr_avg_hist'] = data['ols']['pe_avg_hist'].dropna().rolling(center=False, window=5, min_periods=1).mean()
    for per in [next_per, pers_2]:
        final_val = '%.3f' % (data['ols']['pe_5yr_avg_hist'][period]
                              * (data['ols']['eps'][per]))
        ests.append(("PE", per[1], per[0], final_val))
        if DEBUG or (STOCK_DEBUG and stock == period[1]):
            print("Hist avg PE: {}  Fwd EPS: {}  DV Est {} {}: {}"
                  "".format('%.3f' % (data['ols']['pe_5yr_avg_hist'][period]),
                            '%.3f' % (data['ols']['eps'][per]), per[1], per[0], final_val))

    # P/S
    # Sales per share
    data['ols']['sps'] = data['ols']['revenue'] / data['ols']['weight_avg_shares']
    data['ols']['ps_avg_hist'] = data['ols']['avg_52wk'] / data['ols']['sps']
    data['ols']['ps_curr_hist'] = curr_px / data['ols']['sps']
    data['ols']['ps_fwd'] = ((data['ols']['date_px']
                              * data['is']['weight_avg_shares'])
                             / data['is']['revenue'].shift(1))
    data['ols']['ps_5yr_avg_hist'] = data['ols']['ps_avg_hist'].dropna().rolling(center=False, window=5, min_periods=1).mean()

    for per in [next_per, pers_2]:
        final_val = '%.3f' % (data['ols']['ps_5yr_avg_hist'][period]
                              * (data['ols']['sps'][per]))
        ests.append(("PS", per[1], per[0], final_val))
        if DEBUG or (STOCK_DEBUG and stock == period[1]):
            print("Hist avg PS: {}  Fwd Rev/share: {}  DV Est {} {}: {}"
                  "".format('%.3f' % (data['ols']['ps_5yr_avg_hist'][period]),
                            '%.3f' % (data['ols']['sps'][per]), per[1], per[0], final_val))

    # P/B
    data['ols']['bvps'] = (data['ols']['total_equity']
                           / data['ols']['weight_avg_shares'])
    data['ols']['pb_avg_hist'] = data['ols']['avg_52wk'] / data['ols']['bvps']
    data['ols']['pb_curr_hist'] = curr_px / data['ols']['bvps']
    data['ols']['pb_fwd'] = ((data['ols']['date_px'] * data['is']['weight_avg_shares'])
                             / data['bs']['total_equity'].shift(1))
    data['ols']['pb_5yr_avg_hist'] = data['ols']['pb_avg_hist'].dropna().rolling(center=False, window=5, min_periods=1).mean()

    for per in [next_per, pers_2]:
        final_val = '%.3f' % (data['ols']['pb_5yr_avg_hist'][period]
                              * (data['ols']['bvps'][per]))
        ests.append(("PB", per[1], per[0], final_val))
        if DEBUG or (STOCK_DEBUG and stock == period[1]):
            print("Hist avg PB: {}  Fwd BVPS: {}  DV Est {} {}: {}"
                  "".format('%.3f' % (data['ols']['pb_5yr_avg_hist'][period]),
                            '%.3f' % (data['ols']['bvps'][per]), per[1], per[0], final_val))

    # P/CF
    # cash flow per share
    data['ols']['cfps'] = data['ols']['oper_cf'] / data['ols']['weight_avg_shares']
    data['ols']['pcf_avg_hist'] = data['ols']['avg_52wk'] / data['ols']['cfps']
    data['ols']['pcf_curr_hist'] = curr_px / data['ols']['cfps']
    data['ols']['pcf_fwd'] = ((data['ols']['date_px'] * data['is']['weight_avg_shares'])
                              / data['cf']['oper_cf'].shift(1))
    data['ols']['pcf_5yr_avg_hist'] = data['ols']['pcf_avg_hist'].dropna().rolling(center=False, window=5, min_periods=1).mean()

    for per in [next_per, pers_2]:
        final_val = '%.3f' % (data['ols']['pcf_5yr_avg_hist'][period]
                              * (data['ols']['cfps'][per]))
        ests.append(("PCF", per[1], per[0], final_val))
        if DEBUG or (STOCK_DEBUG and stock == period[1]):
            print("Hist avg PCF: {}  Fwd CF/share: {}  DV Est {} {}: {}"
                  "".format('%.3f' % (data['ols']['pcf_5yr_avg_hist'][period]),
                            '%.3f' % (data['ols']['cfps'][per]), per[1], per[0], final_val))

    # P/FCF
    # free cash flow per share
    data['ols']['fcfps'] = data['ols']['fcf'] / data['ols']['weight_avg_shares']
    data['ols']['pfcf_avg_hist'] = data['ols']['avg_52wk'] / data['ols']['fcfps']
    data['ols']['pfcf_curr_hist'] = curr_px / data['ols']['fcfps']
    data['ols']['pfcf_fwd'] = ((data['ols']['date_px'] * data['is']['weight_avg_shares'])
                               / data['cf']['fcf'].shift(1))
    data['ols']['pfcf_5yr_avg_hist'] = data['ols']['pfcf_avg_hist'].dropna().rolling(center=False, window=5, min_periods=1).mean()

    for per in [next_per, pers_2]:
        final_val = '%.3f' % (data['ols']['pfcf_5yr_avg_hist'][period]
                              * (data['ols']['fcfps'][per]))
        ests.append(("PFCF", per[1], per[0], final_val))
        if DEBUG or (STOCK_DEBUG and stock == period[1]):
            print("Hist avg PFCF: {}  Fwd FCF/share: {}  DV Est {} {}: {}"
                  "".format('%.3f' % (data['ols']['pfcf_5yr_avg_hist'][period]),
                            '%.3f' % (data['ols']['fcfps'][per]), per[1], per[0], final_val))

    # Relative P/E
    # NEED THE EARNIGNS OF THE SNP500
    # data['PE_rel'] = (52WeekAvg * shares) / data['PE_of_SnP']
    # data['PE_rel_curr'] = (cur_px * shares) / data['PE_of_SnP']
    # data['PE_rel_fwd'] = (cur_px * shares) / data['PE_of_SnP'].shift(1)
    # data['PE_rel_5yr_avg'] = PE_rel.rolling(center=False, window=5).mean()
    # for p in [next_per, pers_2]:
        # print("Hist avg PS: {}  Fwd Rev/share: {}  DV Est {} {}: {}"
                # "".format(data['PE_rel__5yr_avg'][period],
        #     data['PE_of_SnP'][p] / data['shares'][period], period[1], period[0]
        #     data['PE_rel__5yr_avg'][period] * data['revenue'][p] / data['shares'][period]))

    # PEG
    # data['PEGY'] = data['PE_avg_hist']
                    #  / ((data['netIncome'].pct_change() + data['divYield']) * 100)
    # data['PEGY_5yr_avg'] = PEGY.rolling(center=False, window=5).mean()
    return data, ests


def ratios_and_valuation(data):
    """
    Add some necessary columns
    """
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
        data[sheet] = data[sheet].reset_index()[data[sheet].reset_index().year
                                                != 'TTM'].set_index(IDX)

    # next qurater est is equal to revenue * average est margin
    # over the last year
    for _ in range(years):
        if hist.empty:
            n_idx = list(data['is'].iloc[-1].name)
        else:
            n_idx = list(hist.iloc[-1].name)
        n_idx = get_next_year(n_idx)
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
            for sheet in ['is', 'bs', 'cf', 'fr']:
                try:
                    val = data[sheet][cat].dropna()
                    data_ols[cat] = data[sheet][cat]
                    x_val = val.reset_index()[['year', 'month']]
                    break
                except KeyError:
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
    hist['net_inc'] = pd.concat([data['is']['net_inc'],
                                 hist['net_inc'].dropna()])
    return hist


def model_est(cum, margin):
    """
    Model based on quarterly estimats
    """
    # some cleanup
    margin = margin.reset_index()[margin.reset_index().date !=
                                  'TTM'].set_index(IDX)
    cum = cum.reset_index()[cum.reset_index().month != ''].set_index(IDX)

    # next qurater est is equal to rev * avg est margin over the last year
    for _ in range(4):
        n_idx = list(cum.iloc[-1].name)
        n_idx = get_next_quarter(n_idx)
        n_data = n_idx + list(margin[-5:-1].mean())
        t_df = pd.DataFrame(dict((key, value) for (key, value) in
                                 zip(IDX+list(margin.columns), n_data)),
                            columns=IDX+list(margin.columns), index=[0]).set_index(IDX)
        margin = margin.append(t_df)

        n_cum_dict = {k: v for k, v in zip(IDX, n_idx)}
        n_cum_dict['revenue'] = cum[-5:-1]['revenue'].mean()

        #########
        # Use mean of previous few years to get there
        #########
        for col in ['cogs', 'rd', 'sga', 'grossProfit', 'mna', 'otherExp']:
            n_cum_dict[col] = (margin.loc[tuple(n_idx)][col]
                               * n_cum_dict['revenue'])
        n_cum_dict['operatingCost'] = (n_cum_dict['rd'] + n_cum_dict['sga']
                                       + n_cum_dict['mna'] + n_cum_dict['otherExp'])
        # operating income
        n_cum_dict['EBIT'] = (n_cum_dict['revenue']
                              - n_cum_dict['operatingCost'] - n_cum_dict['cogs'])
        # Need to update these when we do balance sheet
        total_debt = cum.iloc[-1]['totalLiab']
        cash_and_inv = cum.iloc[-1]['totalCash']
        # 0.6 = about corp rate, 0.02 = about yld on cash, 0.25 = 1/4 of the yr
        n_cum_dict['intExp'] = (total_debt * 0.25 * 0.7
                                - cash_and_inv * 0.25 * 0.02)
        # just assume avg of last yr, gonna be very specific company to company
        n_cum_dict['otherInc'] = cum[-5:-1]['otherInc'].mean()
        n_cum_dict['EBT'] = (n_cum_dict['EBIT'] - n_cum_dict['intExp']
                             + n_cum_dict['otherInc'])
        # average tax rate of the last year
        n_cum_dict['taxes'] = ((cum[-5:-1]['taxes'] / cum[-5:-1]['EBT']).mean()
                               * n_cum_dict['EBT'])
        n_cum_dict['netIncome'] = n_cum_dict['EBT'] - n_cum_dict['taxes']

        # Assume change over the last year continues for next quarter
        # may need to adjust this per company
        n_cum_dict['shares'] = (((((cum.iloc[-1]['shares']
                                    / cum.iloc[-5]['shares']) - 1) / 4) + 1)
                                * cum.iloc[-1]['shares'])
        n_cum_dict['sharesBasic'] = (((((cum.iloc[-1]['sharesBasic']
                                         / cum.iloc[-5]['sharesBasic']) - 1) / 4) + 1)
                                     * cum.iloc[-1]['sharesBasic'])

        n_cum_dict['EPS'] = n_cum_dict['netIncome'] / n_cum_dict['shares']
        n_cum_dict['EPSBasic'] = (n_cum_dict['netIncome']
                                  / n_cum_dict['sharesBasic'])

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


def ols_calc(xvals, yvals):
    """
    ordinary least squares calculation
    """
    xvals = [dt.datetime(int(x[0]), int(float(str(x[1]).replace("E", ""))),
                         1).date() for x in xvals.values]
    xvals = get_year_deltas(xvals)
    a_mat = np.vstack([xvals, np.ones(len(xvals))]).T
    slope, yint = np.linalg.lstsq(a_mat, yvals.values)[0]
    return (slope, yint)


def get_price_data(ticks, comps, method='db'):
    """
    Grab data from API, File, or DB
    """
    pxs = pd.DataFrame()
    # Cant find an API I can trust for EOD stock data
    for ind_t in ticks + comps:
        if method == 'api':
            pass
            # start = dt.date(2000, 1, 1).strftime("%Y-%m-%d")
            # end = dt.datetime.today().date().strftime("%Y-%m-%d")
            # url = "https://www.quandl.com/api/v1/datasets/WIKI/{0}.csv?column=4&sort_order=asc&trim_start={1}&trim_end={2}".format(ind_t, start, end)
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


def analyze_ests(key, data_df, period, years_fwd=2):
    """
    Analyze the results of all the outputs from the valuation techniques
    """
    val_models = ['Hist Comps', 'DFCF', 'PDV']
    val_weights = {
        'Hist Comps': 0.35,
        'DFCF': 0.50,
        'PDV': 0.15
    }
    per = tuple([period[0], key, data_df[0]['bs'].index.values[0][2]])
    print("Tick: {}   Date: {} {}".format(key, per[0], per[2]))
    print("Current Price: {}".format(data_df[0]['ols']['date_px'][per]))
    try:
        data_df[0]['bs'].loc[per]
    except KeyError:
        # Company may havent reported yet in current year
        per = tuple([str(int(period[0])-1), key,
                     data_df[0]['bs'].index.values[0][2]])

    for ind_y in range(1, years_fwd+1):
        year = str(int(per[0]) + ind_y)
        year_est = {}
        for mod in val_models:
            mod_est = []
            for val in VALUATIONS[mod]:
                try:
                    estimate = [float(est[3]) for est in data_df[1] if est[0] == val and
                                est[1] == key and est[2] == year][0]
                except KeyError:
                    # Might not have this model for this year
                    continue
                except IndexError:
                    # Might not have year for this model
                    continue
                mod_est.append(estimate)
                print("Model: {}  tick: {}  year: {}  EST: {}"
                      "".format(val, key, year, estimate))
            if not mod_est:
                continue
            year_est[mod] = sum(mod_est)/len(mod_est)
            print("Models AVG: {}  tick: {}  year: {}  EST: {}"
                  "".format(mod, key, year, '%.4f' % year_est[mod]))
            prem_disc = (year_est[mod] / data_df[0]['ols']['date_px'][per]) - 1
            # Divide by beta
            risk_adj = (((year_est[mod]
                          / data_df[0]['ols']['date_px'][per]) - 1)
                        / data_df[0]['ols']['beta'][per])
            print("Prem/Disc to Current PX: {}  Risk Adj Prem/Disc: {}"
                  "".format('%.4f' % prem_disc, '%.4f' % risk_adj))

        # Assume 50% DFCF, 35% Hist comparables, 15% for peer derived
        year_avg_est = 0
        if len(list(year_est.keys())) < 3:
            # dont have values for all models
            continue
        for yr_key, estimate in year_est.items():
            year_avg_est += estimate * val_weights[yr_key]
        print("Current Price: {}".format(data_df[0]['ols']['date_px'][per]))
        print("Weighted AVG Estimate   tick: {}  year: {}  EST: {}"
              "".format(key, year, '%.4f'% year_avg_est))
        prem_disc = (year_avg_est
                     / data_df[0]['ols']['date_px'][per]) - 1
        # Divide by beta
        risk_adj = (((year_avg_est / data_df[0]['ols']['date_px'][per])
                     - 1) / data_df[0]['ols']['beta'][per])
        print("Prem/Disc to Current PX: {}  Risk Adj Prem/Disc: {}"
              "".format('%.4f' % prem_disc, '%.4f' % risk_adj))


def valuation_model(ticks, mkt, ind, mode='db'):
    """
    Main method for valuation model
    """
    pdb.set_trace()
    full_data = {}
    other = [mkt, ind]
    # Get Historical Price data
    hist_px = get_price_data(ticks, other, mode)
    stock = ticks[0]
    peers = ticks[1:]
    for ind_p in peers:
        # check if we have prices
        if ind_p not in list(hist_px.index.levels[0].values):
            peers.remove(ind_p)
    ticks = [stock] + peers
    print("CURRENT STOCK:  {}".format(stock))

    for ind_t in ticks:
        print("--- running calcs for {} ---".format(ind_t))
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

        data = remove_empty_cols(data)
        data = replace_needed_cols(data)

        # This is only if we have quarterly data
        # data_cum = dataCumColumns(data)
        # data_chg = period_chg(data)
        # data_margin = margin_df(data['is'])[['gross_profit', 'cogs', 'rnd',
        #                                      'sga', 'restruct_mna',
        #                                      'prov_inc_tax', 'other_oper_exp']]

        # Add some columns and adjustemnts and calculate ratios for Use later
        data = ratios_and_valuation(data)

        # project out data based on historicals using OLS regression
        data['ols'] = model_est_ols(10, data)

        # get price info
        data = match_px(data, hist_px, ind_t)
        if DEBUG or (STOCK_DEBUG and ind_t == stock):
            date_px = data['ols']['date_px'].dropna().iloc[-1]
            date_px_idx = data['ols']['date_px'].dropna().index[-1]
            last_px = hist_px.loc[stock].iloc[-1]['px']
            last_px_date = hist_px.loc[stock].iloc[-1].name.strftime("%Y-%m-%d")
            print("PRICE AT LAST STATEMENT ({}-{}):  {}"
                  "".format(date_px_idx[0], date_px_idx[2], date_px))
            print("LAST PRICE ({}):  {}".format(last_px_date, last_px))
        data = get_beta(data, hist_px, ind_t, other[0], other[1])

        period = [i for i in data['is'].index.values if "E" not in i[2]][-1]
        # Get Historical Ratios for valuation
        hist_rats, ests = historical_ratios(data, period, hist_px, stock)
        # Discounted Free Cash Flow Valuation
        dfcf, ests = discount_fcf(hist_rats, period, ests, stock)
        full_data[ind_t] = [dfcf, ests]

    # calculate performance metrics based on price
    px_df = price_perf_anal(period, mkt, ind, hist_px)
    for idx, px_tick in px_df.iterrows():
        if DEBUG or (STOCK_DEBUG and idx == stock):
            print("{} for year: {}  Return: {}  Rel Mkt Ret: {},  Rel Bmk Ret: {}"
                  "".format(idx[0], idx[1], '%.3f' % px_tick['ytd_chg'],
                            '%.3f' % px_tick['mkt_rel_perf'], '%.3f' % px_tick['ind_rel_perf']))

    # Comparisons
    comp_anal = comparison_anal(full_data, period)
    for idx, ind_ca in comp_anal.iterrows():
        if DEBUG or (STOCK_DEBUG and idx[0] == stock):
            if idx[1] == 'net_inc_g':
                print("NET INCOME GROWTH for {}:  {}".format('2018', '%.3f' % ind_ca['2018']))
                print("NET INCOME AVG 5Y GROWTH:  {}".format('%.3f' % ind_ca['avg_5y']))
                print("NET INCOME GROWTH for {}:  {}".format('2019', '%.3f' % ind_ca['2019']))
                print("NET INCOME GROWTH for {}:  {}".format('2020', '%.3f' % ind_ca['2020']))
            if idx[1] == 'revenue_g':
                print("REVENUE GROWTH for {}:  {}".format('2018', '%.3f' % ind_ca['2018']))
                print("REVENUE AVG 5Y GROWTH:  {}".format('%.3f' % ind_ca['avg_5y']))
                print("REVENUE GROWTH for {}:  {}".format('2019', '%.3f' %ind_ca['2019']))
                print("REVENUE GROWTH for {}:  {}".format('2020', '%.3f' % ind_ca['2020']))

    # Peer Derived Value
    full_data, pdv = peer_derived_value(full_data, period)
    for idx, ind_pdv in pdv.iterrows():
        if DEBUG or (STOCK_DEBUG and idx[0] == stock):
            print("{} Peer Derived Price (2019):  {}"
                  "".format(PDV_MAP[idx[1]], ind_pdv['pdv_price_2019']))
            print("{} Peer Derived Price (2020):  {}"
                  "".format(PDV_MAP[idx[1]], ind_pdv['pdv_price_2020']))

    # Analysis of all valuations
    for key, data_df in full_data.items():
        if DEBUG or (STOCK_DEBUG and key == stock):
            analyze_ests(key, data_df, period)


def read_analysis():
    """
    read through inputs file for analysis
    """
    anal_df = pd.read_csv('static/input/analysis.csv', header=None)
    anal_df.columns = ['tick', 'mkt', 'ind', 'peers']
    return anal_df.set_index('tick')


def run_eq_valuation(ticks):
    """
    Given ticks, read analysis input for other parameters
    then run thru equity valuation based on those inputs
    """
    read_val = read_analysis()
    inputs = read_val[read_val.index.isin(ticks.reset_index()['tick'].values)]
    for idx, vals in inputs.iterrows():
        valuation_model([idx]+vals['peers'].strip().split(" "),
                        vals['mkt'].strip(), vals['ind'].strip())


if __name__ == '__main__':
    # use this to pic your analysis setup
    ANAL_ROW = 0
    READ_VAL = read_analysis().iloc[ANAL_ROW]
    # valuation_model(['MSFT'])
    # valuation_model(['MSFT', 'AAPL', 'CSCO', 'INTC', 'ORCL'])
    # valuation_model(['MSFT', 'INTC'])
    valuation_model([READ_VAL[0]] + READ_VAL[3].strip().split(" "),
                    READ_VAL[1], READ_VAL[2])
    