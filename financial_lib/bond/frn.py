import sys
sys.path.append("/home/ubuntu/workspace/finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import datetime, pdb
from app import app
from app.bond.bond import Bond
from app.utils.fi_funcs import *

class FRN(Bond):
    """This class will hold all the variables associated with a Floating Rate Note"""
    
    
    def __init__(self, cusip, issue_dt, mat_dt, sec_type, cpn=0, trade_dt=datetime.date.today(),
                dcc="ACT/ACT", par=100, price=None, ytm=None, pay_freq=0.5,  quoted_sprd=0,
                reset='arrears', first_pay_dt=None, index_freq=None, index="tsy"):
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
        cpn : float
            coupon rate of the bond, expressed in percent terms not dollar amount, DEFAULT = 0
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
        reset : str
            when the reset will take place: "now" or "arrears", arrears meaning the coupon was set at the end
            of the previous period, now meaning the coupon is set on the coupon date
            DEFAULT = "arrears"
        first_pay_dt : str
            first payment date, need this as some bonds have a short stub period before first payment
            instead of a full accrual period, DEFAULT = None
        index_freq : float
            what reference on the curve the payment resets to, important as it defines the
            reference rate for the discount yield
            DEFAULT = pay_freq
        index : str
            index the bond uses, i.e. what curve to go to
            DEFAULT = "tsy"
        quoted_sprd : float
            the spread over the reference rate provided to the bond, set at the issue of the bond
            NOTE - will come in as percent value and divided by 100, ex 2% / 100 = 0.02
            DEFAULT = 0, makes sense for trasury floaters
            
        
        Return
        ======
        NONE
        '''
        super().__init__(cusip, issue_dt, mat_dt, sec_type)
        index_freq = pay_freq if not index_freq else index_freq
        self._dcc = dcc or "ACT/ACT"
        self._par = par
        self._trade_dt = trade_dt
        self._reset = reset
        self._pay_freq = pay_freq
        self._index_freq = index_freq
        self._index = index
        self._quoted_sprd = quoted_sprd / 100
        pdb.set_trace()
        self._bm = self.findBenchmarkRate(ref_dt=self._trade_dt+datetime.timedelta(365*self._index_freq))
        self._cpn = cpn / 100 if cpn else (self._bm[1] + self._quoted_sprd) / 100
        self._ref_rate = self._cpn - self._quoted_sprd      # This is the rate of the index at last reset
        
        
        self._pv = price
        
        if first_pay_dt:
            self._first_pay_dt = datetime.date(int(first_pay_dt[0:4]), int(first_pay_dt[5:7]), int(first_pay_dt[8:10]))
            self._cash_flows = createCashFlows(self._first_pay_dt, self._pay_freq, self._mat_dt, self._cpn, self._par)
            self._cash_flows.insert(0, (self._first_pay_dt, self._cpn * self._par * self._pay_freq))
        else:
            self._cash_flows = createCashFlows(self._issue_dt, self._pay_freq, self._mat_dt, self._cpn, self._par)
        
        if self._pv:
            # Need to code this up
            # self._disc_yld = self.calcDiscountYield()
            pass
        else:
            self._disc_yld = ytm / 100 if ytm else self._bm[1]
            self._pv = self.calcPresentValue()
        
        self._eff_sprd = self.calcEffectiveSprd()
        
    def calcPresentValue(self):
        # Not sure how correct this is
        pdb.set_trace()
        r_adj = self._disc_yld * self._pay_freq
        # used to figure out how far into the pay period we are
        days_to_payment_ratio = (((self._cash_flows[0][0] - self._trade_dt).days)/365) / self._pay_freq
        return ((self._par * self._cpn * self._pay_freq) / (1 + r_adj)**(days_to_payment_ratio) + 
                (self._par / (1 + r_adj)**(days_to_payment_ratio)))
    
    def calcDiscountMargin(self):
        ytm = calcYieldToDate(self._pv, self._par, self._mat_dt, self._cpn, freq=self._pay_freq, start_date=self._trade_dt)
        dm = ytm - self._ref_rate
        #     # also known as effective yield
        #     # http://help.derivativepricing.com/1707.htm
        #     tenor = (self._mat_dt - self._trade_dt).days / 365.25 # assumes 365.25 days in a year
        #     freq = float(self._pay_freq)
        #     # guess ytm = coupon rate, will get us in the ball park
        #     guess = self._cpn
        #     # convert cpn from annual rate to actual coupon value recieved
        #     coupon = self._cpn * self._freq * self._par
        #     cfs = fi_funcs.createCashFlows(self._trade_dt, freq, self._mat_dt, self._cpn, self._par)
        #     # filters for only cash flows that haven't occurred yet
        #     cfs = [c for c in cfs if c[0] > self._trade_dt]
        #     cpn_dts = [((i[0] - self._trade_dt).days / 365, i[1]) for i in cfs]
        
        #     ytm_func = lambda y: \
        #     sum([c/(1+y*freq)**(t/freq) for t,c in cpn_dts]) - self._pv
            
        # # return optimize.newton(ytm_func, guess)
        # return newton_raphson(ytm_func, guess)
        pdb.set_trace()
        return dm
    
    def calcEffectiveSprd(self):
        pass
    
    def calcSimpleMargin(self):
        # https://en.wikipedia.org/wiki/Floating_rate_note
        val = (100 / self._pv) * (((100-self._pv) / years) + self._quoted_sprd)

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # bond = FRN("TEST", "2017-01-01", "2020-01-29", "FRN", cpn=3.5, trade_dt=datetime.date(2017,6,1),
    #             pay_freq=0.5)
    # CFA1 reading 54 - p 422
    bond = FRN("CFA", "2017-01-01", "2022-01-01", "FRN", trade_dt=datetime.date(2017,1,1),
                pay_freq=0.25, quoted_sprd=0.75, cpn=1.85, price=100)
    print(bond.calcDiscountMargin())