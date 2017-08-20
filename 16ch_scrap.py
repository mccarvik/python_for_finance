import sys, pdb
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import datetime as dt

from dx import *

def standard_normal_random_numbers():
    snrn = sn_random_numbers((2, 2, 2), antithetic=False, moment_matching=False, fixed_seed=True)
    print(snrn)
    snrn_mm = sn_random_numbers((2, 3, 2), antithetic=False, moment_matching=True, fixed_seed=True)
    print(snrn_mm)
    print(snrn_mm.mean())
    print(snrn_mm.std())


if __name__ == '__main__':
    standard_normal_random_numbers()