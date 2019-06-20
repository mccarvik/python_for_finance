"""
Screener to filter stocks through the DB
"""
import pdb
import sys
import pandas as pd

sys.path.append("/home/ec2-user/environment/python_for_finance/")
from utils.db_utils import DBHelper

class Filter():
    """
    class to hold different types of filters
    """
    def __init__(self, col, oper, value):
        self.col = col
        self.oper = oper
        self.value = value
        self.create_where_string()

    def create_where_string(self):
        """
        generates the where clause for this filter
        """
        self.clause = self.col + " " + self.oper + " " + self.value


def db_call(table, where_clause, cols):
    """
    Call to DB for data
    """
    with DBHelper() as dbh:
        dbh.connect()
        tick_df = dbh.select(table, where=where_clause, cols=cols)
    return tick_df


def get_tick_info(table, cols, filts):
    """
    Set up the query to call the db and return the possible ticks
    """
    clause = ''
    if filts:
        for ind_filt in filts:
            clause += ind_filt.clause + " and "
        clause = clause[:-5]
    print(clause)
    tick_df = db_call(table, clause, cols)
    return tick_df


def run_filter():
    """
    Creates the filters and runs through to the DB
    """
    # columns we want
    cols = ['tick', 'year', 'month', 'pe_ratio', 'pb_ratio', 'div_yield', 'roe']
    filts = []
    # might get dupes here, need to monitor
    filts.append(Filter('year', "in", "('2018', '2019')"))
    filts.append(Filter('pe_ratio', "<", "20"))
    filts.append(Filter('pe_ratio', ">", "0"))
    filts.append(Filter('pb_ratio', "<", "10"))
    filts.append(Filter('div_yield', ">", "0"))
    filts.append(Filter('market_cap', ">", "0"))
    ticks = get_tick_info('fin_ratios', cols, filts)
    return ticks


def check_momentum(date, ticks):
    """
    Grab the momentum numbers from the given run and filters based on momentum
    """
    mom_df = pd.read_csv("static/output/momentum_returns_{}.csv"
                         "".format(date)).set_index('tick')
    # filter for only the stocks that got through filter
    mom_df = mom_df[mom_df.index.isin(ticks['tick'].values)]
    # filter only for momentum over 70% of peers re Fama-French
    mom_df = mom_df[mom_df.percentile > 0.7]
    return mom_df


def check_big_v_small(date, ticks):
    """
    Grab the small companies that meet other filters
    """
    bvs_df = pd.read_csv("static/output/small_v_big_{}.csv"
                         "".format(date)).set_index('tick')
    # filter for only the stocks that got through filter
    bvs_df = bvs_df[bvs_df.index.isin(ticks['tick'].values)]
    # filter only for smallest 30% of peers re Fama-French
    bvs_df = bvs_df[bvs_df.percentile < 0.3]
    return bvs_df


if __name__ == '__main__':
    TICKS = run_filter()
    check_momentum('20190619', TICKS)
    print(TICKS)
