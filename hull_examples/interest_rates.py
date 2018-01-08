import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

import numpy as np
# import pandas as pd
import datetime as dt


def compounding_conversion(r, yrs, prds=1, cont=False):
    if not cont:
        return (1 + r/prds)**(prds*yrs)
    else:
        return np.exp(r*yrs)

def convert_to_continuous(r, prds):
    return prds * np.log(1 + r/prds)


def convert_from_continuous(r, prds):
    return prds * (np.exp(r/prds) - 1)

# Calculates the bond yield for a bond to equal its market price
def calc_bond_yield(px, settle_dt, mat_dt, cpn, prds):
    pass

# Calculates the coupon a bond would need to trade at par given a yield curve
def calc_par_yield(par, settle_dt, mat_dt, prds):
    pass
    


if __name__ == '__main__':
    # print(compounding_conversion(.10, 2, 2))
    # print(compounding_conversion(.10, 2, cont=True))
    # print(convert_to_continuous(.10, 2))
    # print(convert_from_continuous(.08, 4))
    print(calc_bond_yield(98.39, dt.datetime(2015,1,1), dt.datetime(2017,1,1), 6, 2))
    print(calc_par_yield(100, dt.datetime(2015,1,1), dt.datetime(2017,1,1), 2))