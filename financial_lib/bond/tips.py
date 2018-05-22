import sys
sys.path.append("/home/ubuntu/workspace/finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import datetime, pdb
from app import app
from app.bond.bond import Bond
from app.utils.fi_funcs import *

class TIPS(Bond):
    """This class will hold all the variables associated with a Floating Rate Note"""
    
import sys
sys.path.append("/home/ubuntu/workspace/finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import datetime, pdb
from app import app
from app.bond.bond import Bond
from app.utils.fi_funcs import *

class FRN(Bond):
    """This class will hold all the variables associated with a Floating Rate Note"""
    
    
    def __init__(self, cusip, issue_dt, mat_dt, sec_type, cpn_sprd=0, trade_dt=datetime.date.today(),
                dcc="ACT/ACT", par=100, price=None, ytm=None, pay_freq=0.5, reset_freq=None,
                reset='arrears', first_pay_dt=None, index=None):
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
        cpn_sprd : float
            fixed coupon spread over the inflation index of the bond, expressed in percent terms not dollar amount, DEFAULT = 0
            NOTE - will come in as percent value and divided by 100, ex 2% / 100 = 0.02
        dcc : str
            day count convention, DEFAULT = "30/360"
        par : float
            par value of the bond, DEFAULT = 100
        price : float
            current price of the bond
        ytm : float
            yield to maturity of the bond
            NOTE - will come in as percent value and divided by 100, ex come in as 2(%) and become / 100 = 0.02
        trade_dt : date
            day the calculation is done from, DEFAULT = today
        pay_freq : float
            payment frequency of the bond, expressed in fractional terms of 1 year, ex: 0.5 = 6 months
            DEFAULT = 0.5
        reset_freq : float
            reset frequency of the bond, expressed in fractional terms of 1 year, ex: 0.5 = 6 months
            how often the coupon of the floating rate bond will be reset
            DEFAULT = NONE, but set equal to pay_freq
        reset : str
            when the reset will take place: "now" or "arrears", arrears meaning the coupon was set at the end
            of the previous period, now meaning the coupon is set on the coupon date
            DEFAULT = "arrears"
        first_pay_dt : str
            first payment date, need this as some bonds have a short stub period before first payment
            instead of a full accrual period, DEFAULT = None
        index : float
            what reference on the curve the payment resets to, important as it defines the
            reference rate for the discount yield
            DEFAULT = pay_freq
            
        
        Return
        ======
        NONE
        '''
        super().__init__(cusip, issue_dt, mat_dt, sec_type)
        reset_freq = pay_freq if not reset_freq else reset_freq
        index = pay_freq if not index else index
        self._dcc = dcc or "ACT/ACT"
        self._cpn_sprd = cpn_sprd / 100 if cpn_sprd else 0
        self._par = par
        self._trade_dt = trade_dt
        self._reset = reset
        self._pay_freq = pay_freq
        self._reset_freq = reset_freq
        self._index = index
        self._bm = self.findBenchmarkRate(ref_dt=self._trade_dt+datetime.timedelta(365*self._index))
        self._pv = price