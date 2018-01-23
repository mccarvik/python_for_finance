import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
# sys.path.append("/usr/local/lib/python3.4/dist-packages")
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

from dx import *
from utils.utils import *

import numpy as np
# import pandas as pd
import datetime as dt


yields = [0.01, 0.05, 0.04, 0.01]
dates = [dt.datetime(2015,1,1), dt.datetime(2015,4,1), dt.datetime(2015,10,1), dt.datetime(2017, 1, 1)]
dsr = deterministic_short_rate('dsr', list(zip(dates, yields)))