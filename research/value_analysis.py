import sys
import datetime as dt
from dateutil.relativedelta import relativedelta
sys.path.append("/home/ubuntu/workspace/python_for_finance")
from research.eq_analysis_kelleher.res_utils import get_ticker_info
from research.utils.eod_px_util import FILE_PATH, FILE_NAME


def get_bs_info(ticks=[]):
    if not ticks:
        with open(FILE_PATH + FILE_NAME, "r") as file:
            for line in file:
                ticks.append(line.strip())
    
    import pdb; pdb.set_trace()
    bs_df = get_ticker_info(ticks, 'morningstar')
    # total equity comes in as a percentage
    bs_df = bs_df[['shares', 'totalEquity', 'totalAssets']]
    bs_df['BVPS'] = (bs_df['totalEquity'] * bs_df['totalAssets']) / bs_df['shares']
    bs_df['marketCap'] = bs_df['currentPrice'] * bs_df['shares']
    return 


if __name__ == '__main__':
    # END_DT = dt.datetime.today()
    END_DT = dt.datetime(2019, 2, 22)
    START_DT = END_DT - relativedelta(days=367)
    BS_DF = get_bs_info(['MSFT'])
    print(BS_DF)