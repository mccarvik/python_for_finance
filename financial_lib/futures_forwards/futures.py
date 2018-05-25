import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
import numpy as np
import pandas as pd
import datetime as dt

from utils.fi_funcs import *
from dx.frame import deterministic_short_rate, get_year_deltas


class Futures:
    """class to define the attributes and calcs of a 
    forward or futures object"""
    
    def __init__(self, trade_dt=dt.date.today(), fut_for='for', k=None, ir=0, mat_dt=1, spot=None):
        self._type = fut_for
        # ir should be the risk free rate to this contracts maturity
        self._ir = ir
        self._mat_dt = mat_dt
        self._trade_dt = trade_dt
        self._spot = spot
        if not k:
            self._k = self.calcPrice(self._trade_dt)
        else:
            self._k = k
        
            

    

    def calcPrice(self, d):
        """
        _price is the original price paid for the forward / futures
        This function calculates it for initial purchase. It is static
        for calculations after the origin of the contract
        """
        pdb.set_trace()
        T = get_year_deltas([d, self._mat_dt])[-1]
        return self._spot * (1 + self._ir)**T
        
            
    def calcValue(self, d, spot, r):
        """
        t = time to maturity
        r = interest rate incase rate has changed since time of purchase
        und = underlying value incase und has changed since time of purchase
        ^^^^ all of these maybe useful later
        """
        pdb.set_trace()
        T = get_year_deltas([d, self._mat_dt])[-1]
        return spot - self._k / (1 + r)**T
        

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # f = Futures(ir=0.0825, trade_dt=dt.date(2018, 1, 1), mat_dt=dt.date(2023, 1, 1), spot=72.50)
    f = Futures(ir=0.0825, trade_dt=dt.date(2018, 1, 1), mat_dt=dt.date(2019, 1, 1), k=105)
    print(f.calcValue(dt.date(2018, 4, 2), 102, 0.05))