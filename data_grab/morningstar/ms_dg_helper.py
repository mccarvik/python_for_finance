from collections import defaultdict

COL_MAP = {
    "Gross Margin %" : "grossMargin",                               # Margin
    "Operating Margin %" : "operatingMargin",                       # Margin
    "Dividends" : 'dividendPerShare',                           # Per Share
    "Revenue" : 'revenue',                                  # Gross
    # Here to "EBT Margin" is represented as % of Sales
    "COGS" : "cogs",                                                # Margin
    "SG&A" : "sga",                                                # Margin
    "R&D" : "rd",                                                  # Margin
    "Other" : "other", # should really be 'otherExp'                                      # Margin 
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



DAY_COUNTS = ["daysSalesOutstanding", "daysInv", "payablesPeriod", "cashConvCycle", 
            "recievablesTurnover", 'invTurnover', 'fixedAssetsTurnover']
BALANCE_SHEET = ["EBT", "totalAssets", "cashAndShortTermInv", "accountsRecievable", "inventory", "otherCurrentAssets",
                "totalCurrentAssets", "netPPE", "intangibles", "otherLongTermAssets", 
                "accountsPayable", "shortTermDebt", "taxesPayable", "accruedLiabilities",
                "otherShortTermLiabilities", "totalCurrentLiabilities", "longTermDebt", 
                "otherLongTermLiabilities", "totalLiabilities", "totalEquity",
                "revenue", "netIncome", "trailingEPS"]
INCOME_AND_CASH_FLOW = ["grossProfit", "enterpriseValue", "cogs", "sga", "rd", "other", "operatingIncome", "operatingCashFlow",
                    "capSpending", "freeCashFlow", "workingCapital"]
PER_SHARE = ["bookValuePerShare", "freeCashFlowPerShare", "revenuePerShare",
            "dividendPerShare"]
RATIOS = ["sharpeRatio", "currentRatio", "quickRatio", "financialLeverage", "debtToEquity", 
        "interestCoverage", "capExToSales", "freeCashFlowToSales", "freeCashFlowToNetIncome",
        "trailingPE", "priceToBook", "priceToSales", "pegRatio", "assetTurnoverRatio",
        "treynorRatio", 'priceToCashFlow']
MARGINS = ["grossMargin", "operatingMargin", "netInterestOtherMargin", "EBTMargin",
            "netIncomeMargin", "enterpriseToRevenue"]
RETURNS = ["returnOnAssets", "returnOnEquity", "returnOnCapital", "1yrReturn",
            "3yrReturn", "5yrReturn", "10yrReturn", "52WeekLow", "52WeekHigh",
            "ytdReturn", "50DayMvgAvg", "200DayMvgAvg"]
GROWTH = ["operatingCashFlowGrowth", "freeCashFlowGrowth", "revenueGrowth", "epsGrowth"]
OTHER = ['shares', "payoutRatio", "taxRate", "marketCapital"]
INDEX = ['date', 'ticker']
KEY_STATS = ['currentPrice', "divYield", 'volatility', 'beta', 'marketCorr']

# NOTES:
# Below columns in Mil of local currency (usually USD)
# revenue, operatingIncome, dividendPerShare, bookValuePerShare,
# operatingCashFlow, capSpending, freeCashFlow, workingCapital
''' COLUMNS NOT USED:
Revenue %
Year over Year
3-Year Average
5-Year Average
10-Year Average
Operating Income %
Year over Year
3-Year Average
5-Year Average
10-Year Average
Net Income %
Year over Year
3-Year Average
5-Year Average
10-Year Average
EPS %
Year over Year
3-Year Average
5-Year Average
10-Year Average
'''


remove_strs = ['Mil', 'ZAR', 'USD', 'AUD', 'CAD', 'EUR', 'GBP', 'RUB', 'THB']
# Not provided by morningstar
remove_ticks_ms = ['ADPT', 'AMSG', 'BBCN', 'CMN', 'CLNY', 'CSAL', 'CSC', 'FNBC', 
                    'FTI', 'GMT', 'HWAY', 'HMPR', 'IMS', 'ISLE', 'IILG', 'LMCA',
                    'LMCK', 'MBVT', 'PCCC', 'PSG', 'SWHC', 'SSS', 'TASR', 'TCB', 
                    'SYRG', 'SYUT', 'TSRA', 'TKAI', 'UA.C,', 'USMD', 'MESG', 'TMX.']
# Can't get from yahoo / google datareader
remove_ticks_dr = ['AVG', 'AHS', 'BTX', 'CARO', 'CCF', 'CDK', 'CIX', 'EPM', 'FSP',
                    'HALO', 'IDI', 'IMH', 'LTS', 'LBY', 'LMT', 'MFS', 'NERV', 'NHC',
                    'NWL', 'NUS', 'NBL', 'TIS', 'PRL', 'REI', 'PIP', 'SGA', 'SEB',
                    'FLOW', 'SYN', 'WEX']