import sys
sys.path.append("/home/ubuntu/workspace/finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import datetime, pdb
from app import app
from app.bond.bond import Bond
from app.utils.fi_funcs import *
from dx.frame import *

class Bill(Bond):
    """This class will hold all the variables associated with a Bill
    Bills do not have coupons and are sold at a discount to par"""
    
    def __init__(self, cusip='TEST', trade_dt=datetime.date.today(), sec_type='BILL', mat_dt=datetime.datetime.now()+datetime.timedelta(days=365),
                dcc="30/360", par=100, price=None, dr=None, first_pay_dt=None):
        ''' Constructor
        Parameters
        ==========
        cusip : str
            cusip of this bond
        trade_dt : datetime
            time of trade
        sec_type : str
            security type of the bond
        mat_dt : datetime
            maturity date of the bond
        dcc : str
            day count convention, DEFAULT = "30/360"
        par : float
            par value of the bond, DEFAULT = 100
        price : float
            current price of the bond
        dr : float
            discount rate, not add on rate aka yield to maturity of the bill
            NOTE - will come in as percent value and divided by 100, ex come in as 2(%) and become / 100 = 0.02
        
        Return
        ======
        NONE
        '''
        super().__init__(cusip, trade_dt, mat_dt, sec_type)
        self._dcc = dcc
        self._par = par
        # self._bm = self.findBenchmarkRate()
        self._pv = price
        if self._pv:
            self._disc_rate = self.calcDiscountRate()
            self._addon_rate = self.calcAddOnRate()
        else:
            self._disc_rate = dr / 100 if dr else self._bm[1]
            self._pv = self.calcPresentValue()
            self._addon_rate = self.calcAddOnRate()
    
    def calcDiscountRate(self):
        ''' Discount Rate --> interest rate used to calculate a present value
            Uses a 30/360 calendar by convention, note: (FV - PV) / FV
            FV = future value, PV = present value
            
            Parameters
            ==========
            NONE
            
            Return
            ======
            the calculated discount rate in decimal terms 
        '''
        disc = (self._par - self._pv) / self._par
        days_to_mat = (self._mat_dt - self._trade_dt).days
        pdb.set_trace()
        return ((360 / days_to_mat) * disc)
    
    def calcAddOnRate(self):
        ''' Add on rate 
            Uses a ACT/365 calendar by convention, note: (FV - PV) / PV
            FV = future value, PV = present value
            
            Parameters
            ==========
            NONE
            
            Return
            ======
            the calculated add on rate in decimal terms
        '''
        disc = (self._par - self._pv) / self._pv
        days_to_mat = (self._mat_dt - self._trade_dt).days
        return ((365 / days_to_mat) * disc)
        
    
    def calcPresentValue(self):
        # calc'd using discount rate not add on rate
        # http://www.investopedia.com/exam-guide/series-7/debt-securities/compute-treasury-discount-yield.asp
        days_to_mat = (self._mat_dt - self._trade_dt).days
        val = (self._disc_rate * days_to_mat) / 360
        val = (1 - val) * self._par
        return val


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    bond = Bill("TEST", "2017-01-01", "2017-04-02", "Bill", dr=2.25, par=100, trade_dt=datetime.date(2017,1,1))
    print(bond._pv)
    print(bond._disc_rate)
    print(bond._addon_rate)
    
    