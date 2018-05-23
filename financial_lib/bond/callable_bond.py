import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
from bond import Bond

import numpy as np
import pandas as pd
import datetime as dt

from utils.fi_funcs import *
from fixed_rate_bond import FixedRateBond
from dx.frame import deterministic_short_rate, get_year_deltas



class CallableBond(FixedRateBond):
    """This class will hold all the variables associated with a fixed rate bond"""
    
    def __init__(self, cusip='TEST', trade_dt=dt.date.today(), mat_dt=dt.datetime.now()+dt.timedelta(days=365), 
                sec_type='CallableBond', first_pay_dt=None, freq=0.5, cpn=0, dcc="ACT/ACT", par=100, price=None, ytm=None,
                call_dt=[dt.datetime.now()+dt.timedelta(days=365)], call_px=[100]):
        ''' Constructor
        Parameters
        ==========
        cusip : str
            cusip of this bond
        issue_dt : str
            issue date of the bond
        mat_dt : str
            maturity date of the bond
        sec_type : str
            security type of the bond
        first_pay_dt : str
            first payment date, need this as some bonds have a short stub period before first payment
            instead of a full accrual period, DEFAULT = None
        freq : float
            payment frequency of the bond, expressed in fractional terms of 1 year, ex: 0.5 = 6 months
            DEFAULT = 0.5
        cpn : float
            coupon rate of the bond, expressed in percent terms not dollar amount, DEFAULT = 0
            NOTE - will come in as percent value and divided by 100, ex 2% / 100 = 0.02
        dcc : str
            day count convention, DEFAULT = "ACT/ACT"
        par : float
            par value of the bond, DEFAULT = 100
        price : float
            current price of the bond
        ytm : float
            yield to maturity of the bond
            NOTE - will come in as percent value and divided by 100, ex come in as 2(%) and become / 100 = 0.02
        trade_dt : date
            day the calculation is done from, DEFAULT = today
        call_dt : date
            date of potential call of the bond
        call_px : float
            price of the potential call
        
        Return
        ======
        NONE
        '''
        super().__init__(cusip, trade_dt, mat_dt, sec_type, first_pay_dt, freq, cpn, dcc, par, price, ytm)
        self._call_dt = call_dt
        self._call_px = call_px
    
    def yieldToWorst(self):
        ylds = []
        ylds.append(calcYieldToDate(self._pv, self._par, self._mat_dt, self._cpn, freq=self._pay_freq, start_date=self._trade_dt))
        for i in range(len(self._call_dt)):
            ylds.append(calcYieldToDate(self._pv, self._call_px[i], self._call_dt[i], self._cpn * (self._par/self._call_px[i]), freq=self._pay_freq, start_date=self._trade_dt))
        return min(ylds)
        
    def optionAdjustedPx(self):
        # NEED to use black scholes for this
        pass


def cfa_v1_r54():
    call_dt = [dt.datetime(2021, 1, 1), dt.datetime(2022, 1, 1), dt.datetime(2023, 1, 1)]
    call_px = [102, 101, 100]
    bond = CallableBond(trade_dt=dt.datetime(2017, 1, 1), mat_dt=dt.datetime(2024, 1, 1), freq=1, cpn=8, price=105, call_dt=call_dt, call_px=call_px)
    print(bond.yieldToWorst())
    print(bond.optionAdjustedPx())


if __name__ == '__main__':
    cfa_v1_r54()