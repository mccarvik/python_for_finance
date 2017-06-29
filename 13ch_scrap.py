import sys, pdb
sys.path.append('/usr/share/doc')
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'

PATH = '/home/ubuntu/workspace/python_for_finance/png/ch13/'

def multiplication(self):
        return self.a * self.b
        
class sorted_list(object):
    def __init__(self, elements):
        self.elements = sorted(elements)  # sorted list object
    def __iter__(self):
        self.position = -1
        return self
    def __next__(self):
        if self.position == len(self.elements) - 1:
            raise StopIteration
        self.position += 1
        return self.elements[self.position]

class ExampleOne(object):
    pass

class ExampleTwo(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

class ExampleThree(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def addition(self):
        return self.a + self.b

class ExampleFour(ExampleTwo):
    def addition(self):
        return self.a + self.b

class ExampleFive(ExampleFour):
    def multiplication(self):
        return self.a * self.b

class ExampleSix(ExampleFour):
    multiplication = multiplication

class ExampleSeven(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.__sum = a + b
    multiplication = multiplication
    def addition(self):
        return self.__sum

def object_orientation():
    c = ExampleOne()
    print(c.__str__())
    print(type(c))

    c = ExampleTwo(1, 'text')
    print(c.a)
    print(c.b)
    c.a = 100
    print(c.a)
    c = ExampleOne()
    c.first_name = 'Jason'
    c.last_name = 'Bourne'
    c.movies = 4
    print(c.first_name, c.last_name, c.movies)

    c = ExampleThree(10, 15)
    print(c.addition())
    c.a += 10
    print(c.addition())

    c = ExampleFour(10, 15)
    print(c.addition())

    c = ExampleFive(10, 15)
    print(c.addition())
    print(c.multiplication())

    c = ExampleSix(10, 15)
    print(c.addition())
    print(c.multiplication())

    c = ExampleSeven(10, 15)
    print(c.addition())
    print(c._ExampleSeven__sum)
    c.a += 10
    print(c.a)
    print(c.addition())
    print(c._ExampleSeven__sum)
    print(c.multiplication())
    name_list = ['Sandra', 'Lilli', 'Guido', 'Zorro', 'Henry']
    for name in name_list:
        print(name)

    sorted_name_list = sorted_list(name_list)
    for name in sorted_name_list:
        print(name)

    print(type(sorted(name_list)))
    for name in sorted(name_list):
        print(name)

    print(type(sorted_name_list))
    
def discount_factor(r, t):
    ''' Function to calculate a discount factor.
    
    Parameters
    ==========
    r : float
        positive, constant short rate
    t : float, array of floats
        future date(s), in fraction of years;
        e.g. 0.5 means half a year from now
    
    Returns
    =======
    df : float
        discount factor
    '''
    df = np.exp(-r * t)
      # use of NumPy universal function for vectorization
    return df
    
class short_rate(object):
    ''' Class to model a constant short rate object.
    
    Parameters
    ==========
    name : string
        name of the object
    rate : float
        positive, constant short rate
    
    Methods
    =======
    get_discount_factors :
        returns discount factors for given list/array
        of dates/times (as year fractions)
    '''
    def __init__(self, name, rate):
        self.name = name
        self.rate = rate
    def get_discount_factors(self, time_list):
        ''' time_list : list/array-like '''
        time_list = np.array(time_list)
        return np.exp(-self.rate * time_list)
        
class cash_flow_series(object):
    ''' Class to model a cash flows series.
    
    Attributes
    ==========
    name : string
        name of the object
    time_list : list/array-like
        list of (positive) year fractions
    cash_flows : list/array-like
        corresponding list of cash flow values
    short_rate : instance of short_rate class
        short rate object used for discounting
    
    Methods
    =======
    present_value_list :
        returns an array with present values
    net_present_value :
        returns NPV for cash flow series
    '''
    def __init__(self, name, time_list, cash_flows, short_rate):
        self.name = name
        self.time_list = time_list
        self.cash_flows = cash_flows
        self.short_rate = short_rate
    def present_value_list(self):
        df = self.short_rate.get_discount_factors(self.time_list)
        return np.array(self.cash_flows) * df
    def net_present_value(self):
        return np.sum(self.present_value_list())

class cfs_sensitivity(cash_flow_series):
    def npv_sensitivity(self, short_rates):
        sr = short_rate('r', 0.05)
        npvs = []
        for rate in short_rates:
            sr.rate = rate
            npvs.append(self.net_present_value())
        return np.array(npvs)

def short_rate_class():
    t = np.linspace(0, 5)
    for r in [0.01, 0.05, 0.1]:
        plt.plot(t, discount_factor(r, t), label='r=%4.2f' % r, lw=1.5)
    plt.xlabel('years')
    plt.ylabel('discount factor')
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig(PATH + 'short_rate.png', dpi=300)
    plt.close()
    
    sr = short_rate('r', 0.05)
    print(sr.name, sr.rate)
    
    time_list = [0.0, 0.5, 1.0, 1.25, 1.75, 2.0]  # in year fractions
    print(sr.get_discount_factors(time_list))
    
    for r in [0.025, 0.05, 0.1, 0.15]:
        sr.rate = r
        plt.plot(t, sr.get_discount_factors(t),
                 label='r=%4.2f' % sr.rate, lw=1.5)
    plt.xlabel('years')
    plt.ylabel('discount factor')
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig(PATH + 'short_rate2.png', dpi=300)
    plt.close()

    pdb.set_trace()
    sr.rate = 0.05
    cash_flows = np.array([-100, 50, 75])
    time_list = [0.0, 1.0, 2.0]
    disc_facts = sr.get_discount_factors(time_list)
    print(disc_facts)
    print(disc_facts * cash_flows)
    print(np.sum(disc_facts * cash_flows))
    sr.rate = 0.15
    print(np.sum(sr.get_discount_factors(time_list) * cash_flows))
    
    sr.rate = 0.05
    cfs = cash_flow_series('cfs', time_list, cash_flows, sr)
    print(cfs.cash_flows)
    print(cfs.time_list)
    print(cfs.present_value_list())
    print(cfs.net_present_value())

    cfs_sens = cfs_sensitivity('cfs', time_list, cash_flows, sr)
    short_rates = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
    npvs = cfs_sens.npv_sensitivity(short_rates)
    print(npvs)
    plt.plot(short_rates, npvs, 'b')
    plt.plot(short_rates, npvs, 'ro')
    plt.plot((0, max(short_rates)), (0, 0), 'r', lw=2)
    plt.grid(True)
    plt.xlabel('short rate')
    plt.ylabel('net present value')
    plt.savefig(PATH + 'cash_flow.png', dpi=300)
    plt.close()

# class short_rate_g(trapi.HasTraits):
#     name = trapi.Str
#     rate = trapi.Float
#     time_list = trapi.Array(dtype=np.float, shape=(5,))
#     def get_discount_factors(self):
#         return np.exp(-self.rate * self.time_list)

def short_rate_gui():
    sr = short_rate_g()
    sr.name = 'sr_class'
    sr.rate = 0.05
    sr.time_list = [0.0, 0.5, 1.0, 1.5, 2.0]
    print(sr.rate)
    print(sr.time_list)
    print(sr.get_discount_factors())


if __name__ == "__main__":
    # object_orientation()
    short_rate_class()
    # Not compatible for Python3
    # short_rate_gui()