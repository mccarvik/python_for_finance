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
    


if __name__ == '__main__':
    print(compounding_conversion(.10, 2, 2))
    print(compounding_conversion(.10, 2, cont=True))
    print(convert_to_continuous(.10, 2))
    print(convert_from_continuous(.08, 4))
    