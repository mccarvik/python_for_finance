"""
Run through the full analysis of equities
"""
import pdb
import sys

# TODO
# update the ret for the stock universe
# add volatility and sharpe ratio to the calc
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
    ticks = timeme(check_momentum)('20190712', ticks)
    print("{} stocks thru momentum checks\n".format(len(ticks)))

    # check recent big vs small results
    print("Running Big vs. Small Filter:")
    ticks = timeme(check_big_v_small)('20190703', ticks.reset_index())
    print("{} stocks thru big vs small filter\n".format(len(ticks)))

    # remove ticks that should be ignored
    print("Ignoring Certain Symbols:")
    ticks = timeme(ignore_ticks)(ticks.reset_index())
    print("{} stocks thru ignore filter\n".format(len(ticks)))

    # Run equity valuation
    print("Running Equity Valuation on:  {}".format(ticks.index.levels[0].values))
    timeme(run_eq_valuation)(ticks)
    print()


def ignore_ticks(ticks):
    """
    Function to remove certain securities that got through the checks
    but arent good investments
    """
    # might be good - SACH, OBCI, ACU
    ignore_syms = ['CDOR', 'BOTJ', 'DRAD', 'SACH', 'CELP', 'ASFI', 'OBCI', 'ACU']
    ticks = ticks.set_index('tick')
    ticks = ticks[~ticks.index.isin(ignore_syms)]
    ticks = ticks.reset_index().set_index(["tick", "month", "year"])
    return ticks
    


if __name__ == '__main__':
    run_analysis()
