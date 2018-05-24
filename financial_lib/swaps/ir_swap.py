import sys
sys.path.append("/home/ubuntu/workspace/python_for_finance")
import numpy as np
import pandas as pd
import datetime as dt

from utils.fi_funcs import *
from dx.frame import deterministic_short_rate, get_year_deltas

# from app import app
# from app.bond import bond_fixed

class IR_Swap():
    """ Class to represent a fixed for floating swap """
    
    def __init__(self, trade_dt=datetime.date(2015,1,1), mat_dt=dt.datetime.now()+dt.timedelta(days=365), notl=100, 
                fixed_rate=None, fixed_pay_freq=0.5, float_pay_freq=0.5, float_pay_reset=0.5, float_curve=None, dcc_fixed='ACT/ACT'):
        self._mat_dt = mat_dt
        self._trade_dt = trade_dt       # for now
        self._notl = notl
        self._fixed_pay_freq = fixed_pay_freq
        self._fixed_pay_dates = [c[0] for c in createCashFlows(self._trade_dt, self._fixed_pay_freq, self._mat_dt, 0, 100, par_cf=False)]
        self._float_pay_freq = float_pay_freq
        self._float_pay_reset = float_pay_reset
        self._float_ref = float_curve
        self._dcc_fixed = dcc_fixed
        if fixed_rate:
            self._fixed_rate = fixed_rate
        else:
            self._fixed_rate = self.calcFixedRate()
        self._val = self.calcSwapValue(self._trade_dt, self._float_ref)
    
    def calcFixedRate(self):
        B_sum = []
        for d in self._fixed_pay_dates:
            delt = get_year_deltas([self._trade_dt, d])[-1]
            r = self._float_ref.get_interpolated_yields([self._trade_dt, d])[-1][1]
            B_sum.append(1 / (1 + r * delt))
        B0 = B_sum[-1]
        return round(((1 - B0) / sum(B_sum)) / self._fixed_pay_freq, 5)
    
    def calcSwapValue(self, cur_dt, new_curve):
        # Fixed Side Value
        B_sum = []
        for d in [pd for pd in self._fixed_pay_dates if pd > cur_dt]:
            delt = get_year_deltas([cur_dt, d])[-1]
            r = new_curve.get_interpolated_yields([self._trade_dt, d])[-1][1]
            B_sum.append(1 / (1 + r * delt))
        B0 = B_sum[-1]
        fixed_val = (self._fixed_rate * self._fixed_pay_freq * sum(B_sum)) + B0
        
        # Float Side Value
        first_pay_dt = self._fixed_pay_dates[0]
        delt = get_year_deltas([cur_dt, first_pay_dt])[-1]
        r = self._float_ref.get_interpolated_yields([self._trade_dt, first_pay_dt])[-1][1] * self._float_pay_freq
        disc = 1 / (1 + new_curve.get_interpolated_yields([self._trade_dt, first_pay_dt])[-1][1] * delt)
        float_val = (1 + r) * disc
        
        # val from fixed side
        return (fixed_val - float_val) * self._notl
    

def cfa_v2_r50():
    rates = [0.03, 0.0345, 0.0358, 0.0370, 0.0375]
    dates = [dt.datetime(2017,1,1), dt.datetime(2017,4,2), dt.datetime(2017,7,2), dt.datetime(2017,10,1), dt.datetime(2018, 1, 1)]
    dsr = deterministic_short_rate('dsr', list(zip(dates, rates)))
    irs = IR_Swap(trade_dt=dt.datetime(2017,1,1), mat_dt=dt.datetime(2018,1,1), fixed_pay_freq=0.25, float_pay_freq=0.25, float_curve=dsr, notl=30)
    
    # New Curve
    rates = [0.04, 0.0425, 0.0432, 0.0437, 0.0444]
    dates = [dt.datetime(2017,1,1), dt.datetime(2017,4,2), dt.datetime(2017,7,2), dt.datetime(2017,10,1), dt.datetime(2018, 1, 1)]
    dsr = deterministic_short_rate('dsr', list(zip(dates, rates)))
    print(irs.calcSwapValue(dt.datetime(2017, 3, 1), dsr))
    
    
if __name__ == '__main__':
    cfa_v2_r50()