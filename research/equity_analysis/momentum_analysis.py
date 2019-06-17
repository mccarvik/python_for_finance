import sys
import datetime as dt
from dateutil.relativedelta import relativedelta
sys.path.append("/home/ubuntu/workspace/python_for_finance")
from research.utils.eod_px_util import get_db_pxs

def calcMomReturn(px_dfs, trade_dt):
    """
    calculates the 12-2 return, aka return from 12 months ago to one month ago
    """
    start_dt = trade_dt - relativedelta(days=365)
    end_dt = trade_dt - relativedelta(days=30)
    ret_12_2 = px_df.loc[end_dt.strftime('%Y-%m-%d')] / px_df.loc[start_dt.strftime('%Y-%m-%d')] - 1
    ret_12_2 = ret_12_2.dropna().rename(columns={'px': 'ret'})['ret'].sort_values(ascending=False)
    return ret_12_2


if __name__ == '__main__':
    # END_DT = dt.datetime.today()
    END_DT = dt.datetime(2019, 2, 22)
    START_DT = END_DT - relativedelta(days=367)
    # px_df = get_db_pxs(["A", "MSFT"], START_DT, END_DT)
    px_df = get_db_pxs(s_date=START_DT, e_date=END_DT)
    calcMomReturn(px_df, END_DT)
    print(px_df)