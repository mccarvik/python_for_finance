import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
h5 = pd.HDFStore('./vstoxx_data_31032014.h5','r')
fut_data = h5['futures_data']
opt_data = h5['options_data']
h5.close()
opt_data.info()
opt_data[['DATE','MATURITY','TTM','STRIKE','PRICE']].head()

opt_data['IMP_VOL']=0.0
from bsm_functions import *
tol = 0.5
V0 = 17.66
r = 0.01
for opt in opt_data.index:
    fwd = fut_data[fut_data['MATURITY'] == opt_data.loc[opt]['MATURITY']]['PRICE'].values[0]
    if (fwd * (1-tol) < opt_data.loc[opt]['STRIKE'] < fwd * (1+tol)):
        imp_vol = bsm_call_imp_vol(
            V0,
            opt_data.loc[opt]['STRIKE'],
            opt_data.loc[opt]['TTM'],
            r,
            opt_data.loc[opt]['PRICE'],
            sigma_est=2.0,
            it=100)
        opt_data['IMP_VOL'].loc[opt] = imp_vol
opt_data['IMP_VOL']

plot_data = opt_data[opt_data['IMP_VOL'] > 0]
mats = sorted(set(opt_data['MATURITY']))

# this code used only for iPython interpreter
# %matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
for mat in mats:
    data = plot_data[opt_data.MATURITY == mat]
    plt.plot(data['STRIKE'], data['IMP_VOL'],
            label=mat.date(), lw=1.5)
    plt.plot(data['STRIKE'], data['IMP_VOL'], 'r.')
plt.grid(True)
plt.xlabel('strike')
plt.ylabel('implied volatility of volatility')
plt.legend()
plt.savefig('png/book_examples/vol_smile.png', dpi=300)
# plt.show()

keep = ['PRICE', 'IMP_VOL']
group_data = plot_data.groupby(['MATURITY','STRIKE'])[keep]
group_data = group_data.sum()
print(group_data.head())
# print(group_data.index.levels)
