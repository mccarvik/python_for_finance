import numpy as np
import scipy.stats as ss
import time, sys
from app import app
from math import sqrt, pi, log, e

class Futures:
    """class to define the attributes and calcs of a 
    forward or futures object"""
    
    def __init__(self, fut_for='for', und=100, ir=0, cst_cry=0, inc_yld=0, conv_yld=0, t=1, p=None):
        self._type = fut_for
        self._und = und
        self._ir = ir
        self._cst_cry = cst_cry
        self._inc_yld = inc_yld
        self._conv_yld = conv_yld
        self._tenor = t
        self._price = p 
        self.calcMissing()
    
    def calcMissing(self):
        if not self._price:
            self._price = self.calcPrice()
        else:
            # assume if price is filled in, calc value
            self._price = float(self._price)
            self._value = self.calcValue()
    
    def calcPrice(self):
        """
        _price is the original price paid for the forward / futures
        This function calculates it for initial purchase. It is static
        for calculations after the origin of the contract
        """
        
        if self._type == 'for':
            return (self._und) * (e**((self._ir + self._cst_cry - self._conv_yld - self._inc_yld)*self._tenor))
            
    def calcValue(self, r=None, t=None, und=None):
        """
        t = time to maturity
        r = interest rate incase rate has changed since time of purchase
        und = underlying value incase und has changed since time of purchase
        ^^^^ all of these maybe useful later
        """
        # calculate the value of what the future would be today
        cur_value = (self._und) * (e**((self._ir + self._cst_cry - self._conv_yld - self._inc_yld)*self._tenor))
        # Find the difference between todays value and the price paid and discount it to maturity
        return (cur_value - self._price) * (e**(self._tenor*((-1)*self._ir)))
        

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    pass