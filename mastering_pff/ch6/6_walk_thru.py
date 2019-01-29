import pdb
import os.path
import numpy as np
import pandas as pd
import datetime as dt
from urllib.request import urlretrieve
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm

url_path = 'http://www.stoxx.com/download/historical_values/'
stoxxeu600_url = url_path + 'hbrbcpe.txt'
vstoxx_url = url_path + 'h_vstoxx.txt'

# Save file to local target destination.
data_folder = "/home/ubuntu/workspace/python_for_finance/mastering_pff/ch6/"
IMG_PATH = "/home/ubuntu/workspace/python_for_finance/mastering_pff/png/"

stoxxeu600_filepath = data_folder + "stoxxeu600.txt"
vstoxx_filepath = data_folder + "vstoxx.txt"

# grab data
urlretrieve(stoxxeu600_url, stoxxeu600_filepath)
urlretrieve(vstoxx_url, vstoxx_filepath)

print(os.path.isfile(stoxxeu600_filepath))
print(os.path.isfile(vstoxx_filepath))

with open(stoxxeu600_filepath, 'r') as opened_file:
    for i in range(5):
        # print(opened_file.readline())
        pass
        
# Get stoxx 600
columns = ['Date', 'SX5P', 'SX5E', 'SXXP', 'SXXE',
           'SXXF', 'SXXA', 'DK5F', 'DKXF', 'EMPTY']
stoxxeu600 = pd.read_csv(stoxxeu600_filepath,
                 index_col=0,
                 parse_dates=True,                 
                 dayfirst=True,
                 header=None,
                 skiprows=4, 
                 names=columns,
                 sep=';'
                 )  
del stoxxeu600['EMPTY']
# print(stoxxeu600.info())

with open(vstoxx_filepath, 'r') as opened_file:
    for i in range(5):
        # print(opened_file.readline())
        pass

vstoxx = pd.read_csv(vstoxx_filepath,
                 index_col=0, 
                 parse_dates=True, 
                 dayfirst=True,
                 header=2)
vstoxx.info()

# setting the cutoff dates to the same date
cutoff_date = dt.datetime(1999, 1, 4)
data = pd.DataFrame(
{'EUROSTOXX' :stoxxeu600['SX5E'][stoxxeu600.index >= cutoff_date],
 'VSTOXX':vstoxx['V2TX'][vstoxx.index >= cutoff_date]})
data = data.dropna()
data.info()
print(data.head(5))

# data.describe()
# data.plot(subplots=True, figsize=(10, 8), color="blue", grid=True)
# plt.savefig(IMG_PATH + 'hist_levels.png', dpi=300)
# plt.close()

# data.diff().hist(figsize=(10, 5), color='blue', bins=100)
# plt.savefig(IMG_PATH + 'histogram.png', dpi=300)
# plt.close()

# data.pct_change().hist(figsize=(10, 5), color='blue', bins=100)
# plt.savefig(IMG_PATH + 'pct_chg.png', dpi=300)
# plt.close()

log_returns = np.log(data / data.shift(1)).dropna()
# log_returns.plot(subplots=True, figsize=(10, 8), color='blue', grid=True)
# plt.savefig(IMG_PATH + 'log_rets.png', dpi=300)
# plt.close()

# negative correlation between underlying log returns and volatility
print(log_returns.corr())
# log_returns.plot(figsize=(10,8), x="EUROSTOXX", y="VSTOXX", kind='scatter')
# ols_fit = sm.OLS(log_returns['VSTOXX'].values, log_returns['EUROSTOXX'].values).fit()
# plt.plot(log_returns['EUROSTOXX'], ols_fit.fittedvalues, 'r')
# plt.savefig(IMG_PATH + 'ols_fit.png', dpi=300)
# plt.close()

# rolling correlation
pd.rolling_corr(log_returns['EUROSTOXX'], log_returns['VSTOXX'], window=252).plot(figsize=(10,8))
plt.ylabel('Rolling Annual Correlation')
plt.savefig(IMG_PATH + 'rolling_corr.png', dpi=300)
plt.close()