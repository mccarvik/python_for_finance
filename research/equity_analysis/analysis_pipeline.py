"""
Run through the full analysis of equities
"""
import pdb
import sys

# TODO
# update all the loaders
# backtest

sys.path.append("/home/ec2-user/environment/python_for_finance/")
from stock_screener import run_filter, check_momentum, check_big_v_small
from equity_valuation import run_eq_valuation
from utils.helper_funcs import timeme

def run_analysis():
    """
    Function to run thru the whole analysis process
    """
    # Run the filter
    print("Running Screener:")
    ticks = timeme(run_filter)()
    print("{} stocks thru filter\n".format(len(ticks)))

    # check recent momentum returns
    print("Running Momentum Check:")
    ticks = timeme(check_momentum)('20190621', ticks)
    print("{} stocks thru momentum checks\n".format(len(ticks)))

    # check recent big vs small results
    print("Running Big vs. Small Filter:")
    ticks = timeme(check_big_v_small)('20190621', ticks.reset_index())
    print("{} stocks thru big vs small filter\n".format(len(ticks)))

    # Run equity valuation
    # pdb.set_trace()
    print("Running Equity Valuation:")
    timeme(run_eq_valuation)(ticks)
    print()


if __name__ == '__main__':
    run_analysis()
