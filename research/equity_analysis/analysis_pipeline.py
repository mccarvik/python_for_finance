import pdb
import sys

sys.path.append("/home/ec2-user/environment/python_for_finance/")
from stock_screener import run_filter, check_momentum
from equity_valuation
from utils.helper_funcs import timeme

def run_analysis():
    """
    Function to run thru the whole analysis process
    """
    # Run the filter
    pdb.set_trace()
    print("Running Screener:")
    ticks = timeme(run_filter)()
    
    # check recent momentum returns
    pdb.set_trace()
    print("Running Momentum Check:")
    ticks = timeme(check_momentum)('20190619', ticks)
    
    pdb.set_trace()
    print("Running Equity Valuation:")
    timeme(run_eq_valuation)(ticks)
    print()



if __name__ == '__main__':
    run_analysis()