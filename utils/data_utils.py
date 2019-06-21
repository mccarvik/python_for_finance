import sys
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
from utils.db_utils import *
from collections import defaultdict

def getKeyStatsDataFrame(date=datetime.date.today().strftime('%Y-%m-%d'), tickers=None, table='key_stats_yahoo'):    
    ''' Will retrieve the key financial stats from the DB for a given day and tickers
        Will also clean the dataframe data and add any custom columns
    Parameters
    ==========
    date : date
        Date of values retrieved
        DEFAULT = Today
    tickers : list of strings
        list of tickers to be grabbed
        DEFAULT = NONE, will grab everything for given day
    table : string
        The table we are pulling from
        DEFAULT = key_stats_yahoo
    
    Return
    ======
    df : dataframe
        The stats for the given day and tickers
    '''
    where_ticks = "(\""
    if tickers:
        for t in tickers:
            where_ticks += t + "\",\""
    where_ticks = where_ticks[:-2] + ")"
    with DBHelper() as db:
        db.connect()
        if date != '':
            if tickers:
                df = db.select(table, where="date='{0}' and ticker in {1}".format(date, where_ticks))
            else:
                df = db.select(table, where="date='{0}'".format(date))
        else:
            if tickers:
                df = db.select(table, where="ticker in {0}".format(where_ticks))
            else:
                df = db.select(table)
    return df


COL_MAP = {
    "Gross Margin %" : "grossMargin",                               # Margin
    "Operating Margin %" : "operatingMargin",                       # Margin
    "Dividends" : 'dividendPerShare',                           # Per Share
    "Revenue" : 'revenue',                                  # Gross
    # Here to "EBT Margin" is represented as % of Sales
    "COGS" : "cogs",                                                # Margin
    "SG&A" : "sga",                                                # Margin
    "R&D" : "rd",                                                  # Margin
    "Other" : "other",                                              # Margin
    "Net Int Inc & Other" : "netInterestOtherMargin",               # Margin
    "EBT Margin" : "EBTMargin",                                     # Margin
    # 
    "Operating Income" : "operatingIncome",                 # Gross
    "Current Ratio" : "currentRatio",                               # Ratio
    "Quick Ratio" : "quickRatio",                                   # Ratio    
    "Financial Leverage" : "financialLeverage",                     # Ratio    
    "Debt/Equity" : "debtToEquity",                                 # Ratio    
    "Net Income" : "netIncome",                             # Gross
    "Earnings Per Share" : "trailingEPS",                       # Gross
    "Payout Ratio % *" : "payoutRatio",                             # Ratio
    "Shares" : "shares",                                        # Gross
    "Book Value Per Share *" : "bookValuePerShare",             # Per Share
    "Operating Cash Flow" : "operatingCashFlow",            # Gross
    "Cap Spending" : "capSpending",                         # Gross
    "Free Cash Flow" : "freeCashFlow",                      # Gross
    "Free Cash Flow Per Share *" : "freeCashFlowPerShare",      # Per Share
    "Working Capital" : "workingCapital",                   # Gross
    "Tax Rate %" : "taxRate",                                       # Ratio
    "Net Margin %" : "netIncomeMargin",                             # Margin
    "Asset Turnover (Average)" : "assetTurnoverRatio",              # Ratio
    "Return on Assets %" : "returnOnAssets",                        # Ratio
    "Return on Equity %" : "returnOnEquity",                        # Ratio
    "Return on Invested Capital %" : "returnOnCapital",             # Ratio
    "Interest Coverage" : "interestCoverage",                       # Ratio
    "Operating Cash Flow Growth % YOY" : "operatingCashFlowGrowth", # Ratio
    "Free Cash Flow Growth % YOY" : "freeCashFlowGrowth",           # Ratio
    "Cap Ex as a % of Sales" : "capExToSales",                      # Ratio
    "Free Cash Flow/Sales %" : "freeCashFlowToSales",               # Ratio
    "Free Cash Flow/Net Income" : "freeCashFlowToNetIncome",        # Ratio
    # Here to "totalEquity" is represented as % of Total Assets
    "Cash & Short-Term Investments" : "cashAndShortTermInv",        # Ratio
    "Accounts Receivable" : "accountsRecievable",                   # Ratio
    "Inventory" : "inventory",                                      # Ratio
    "Other Current Assets" : "otherCurrentAssets",                  # Ratio
    "Total Current Assets" : "totalCurrentAssets",                  # Ratio
    "Net PP&E" : "netPPE",                                          # Ratio
    "Intangibles" : "intangibles",                                  # Ratio
    "Other Long-Term Assets" : "otherLongTermAssets",               # Ratio
    "Accounts Payable" : "accountsPayable",                         # Ratio
    "Short-Term Debt" : "shortTermDebt",                            # Ratio
    "Taxes Payable" : "taxesPayable",                               # Ratio
    "Accrued Liabilities" : "accruedLiabilities",                   # Ratio
    "Other Short-Term Liabilities" : "otherShortTermLiabilities",   # Ratio
    "Total Current Liabilities" : "totalCurrentLiabilities",         # Ratio
    "Long-Term Debt" : "longTermDebt",                              # Ratio
    "Other Long-Term Liabilities" : "otherLongTermLiabilities",     # Ratio
    "Total Liabilities" : "totalLiabilities",                       # Ratio
    "Total Stockholders' Equity" : "totalEquity",                   # Ratio
    #
    "Days Sales Outstanding" : "daysSalesOutstanding",              # Gross
    "Days Inventory" : "daysInv",                                   # Gross
    "Payables Period" : "payablesPeriod",                           # Gross
    "Cash Conversion Cycle" : "cashConvCycle",                      # Gross
    "Receivables Turnover" : "recievablesTurnover",                 # Gross
    "Inventory Turnover" : "invTurnover",                           # Gross
    "Fixed Assets Turnover" : "fixedAssetsTurnover",                # Gross
}

CUSTOM_COL_MAP = {
    "Current Price" : "currentPrice",                               # Gross
    "Revenue Per Share" : "revenuePerShare",                        # Per Share
    "Total Cash Per Share" : "totalCashPerShare",                   # Per Share
    "Dividend Yield" : "divYield",                                  # Ratio
    "Trailing PE" : "trailingPE",                                   # Ratio
    "Price to Book" : "priceToBook",                                # Ratio
    "Price to Sales" : "priceToSales",                              # Ratio
    "Revenue Growth" : "revenueGrowth",                             # Ratio
    "EPS Growth" : "epsGrowth",                                     # Ratio
    "PEG Ratio" : "pegRatio",                                       # Ratio
    "1 Year Return" : "1yrReturn",                                  # Percent
    "3 Year Return" : "3yrReturn",                                  # Percent
    "5 Year Return" : "5yrReturn",                                  # Percent
    "10 Year Return" : "10yrReturn",                                # Percent
    "52 Week High" : "52WeekHigh",                                  # Gross
    "52 Week Low" : "52WeekLow",                                    # Gross
    "YTD Return" : "ytdReturn",                                     # Percent
    "Gross Profit" : "grossProfit",                                 # Gross
    "Market Capital" : "marketCapital",                             # Gross
    "Enterprise Value" : "enterpriseValue",                         # Gross
    "Total Assets" : "totalAssets",                                 # Gross
    "Enterprise To Revenue" : "enterpriseToRevenue",                # Margin
    "EBT" : "EBT",                                                  # Gross
    "50 Day Moving Average" : "50DayMvgAvg",                        # Gross
    "200 Day Moving Average" : "200DayMvgAvg",                      # Gross
    "Sortino Ratio" : "sortinoRatio",                               # Ratio
    "Downside Volatility" : "downsideVol",                          # Gross
    "Treynor Ratio" : "treynorRatio",                               # Ratio
    "Beta" : "beta",                                                # Gross
    "Market Correlation" : "marketCorr",                            # Gross
    "Price To Free Cash Flow" : "priceToCashFlow"                   # Ratio
}


DAY_COUNTS = ["days_sales_outstanding", "days_of_inv_on_hand", 
              "days_payables_outstanding", 'payables_turnover',
            "receivables_turnover", 'inv_turnover', 'fixed_asset_turnover',
            "asset_turnover", 'cash_conversion_cycle', 'oper_cycle']
PER_SHARE = ["bvps", 'net_income_per_share', "fcf_per_share", "rev_per_share",
             "cash_per_share", "oper_cf_per_share"]
RETURNS = ["roa", "roe", "ret_on_cap", "ret_1y", "ret_2y", "ret_3y", "ret_5y"]
FWD_RETURNS = ['retfwd_1y', 'retfwd_2y', 'retfwd_3y', 'retfwd_5y']
MARGINS = ["gross_prof_marg", "oper_prof_marg", "oper_cf_sales_ratio",
           "pretax_prof_marg", "net_prof_marg", 'ebit_per_rev']
INDEX = ['year', 'tick', 'month']
RATIOS = ["curr_ratio", "pfcf_ratio", 'pcf_ratio', 'pocf_ratio', 
          'cash_ratio', "debt_to_equity", "interest_coverage_ratio",
          "pe_ratio", "pb_ratio", "ps_ratio", "peg_ratio", 'debt_ratio',
          "ev_multiple", "capex_to_rev", "quick_ratio", 'cf_to_debt_ratio']
OTHER = ["div_payout_ratio", "eff_tax_rate", "market_cap", "div_yield", 
         "date_px"]
