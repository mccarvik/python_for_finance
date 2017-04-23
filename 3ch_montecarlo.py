from bsm_functions import bsm_call_value
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Need this to set up modules
import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")

def pure_python():
    # Parameters
    S0 = 100.0
    K=105.0
    T=1.0
    r=0.05
    sigma=0.2
    print(bsm_call_value(S0, K, T, r, sigma))
    
    from time import time
    from math import exp, sqrt, log
    from random import gauss, seed
    
    seed(20000)
    t0 = time()
    
    M = 50
    dt = T / M
    I = 250000
    
    # Simulating I paths with M time steps
    S = []
    for i in range(I):
        path = []
        for t in range(M+1):
            if t==0:
                path.append(S0)
            else:
                z = gauss(0.0, 1.0)
                St = path[t-1] * exp((r-0.5 * sigma ** 2) * dt + sigma * sqrt(dt)*z)
                path.append(St)
        S.append(path)
    
    # Calculating the Monte Carlo Estimator
    C0 = exp(-r * T) * sum([max(path[-1] - K, 0) for path in S]) / I
    
    # Results output
    tpy = time() - t0
    print("European Option value %7.3f" % C0)
    print("Duration in Seconds   %7.3f" % tpy)

def numpy_vectors():
    import numpy as np
    import math
    from time import time
    np.random.seed(20000)
    t0 = time()
    
    # Parameters
    S0 = 100.0
    K=105.0
    T=1.0
    r=0.05
    sigma=0.2
    M = 50
    dt = T / M
    I = 250000
    
    # Simulating I paths with M time steps
    S = np.zeros((M+1, I))
    S[0] = S0
    import pdb; pdb.set_trace()
    for t in range(1, M + 1):
        z = np.random.standard_normal(I) # pseudorandom numbers
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z)
        # vectorized operation per time step over all paths
    
    C0 = math.exp(-r * T) * np.sum(np.maximum(S[-1] - K, 0)) / I
    
    # Results output
    tnp1 = time() - t0
    print("European Option value %7.3f" % C0)
    print("Duration in Seconds   %7.3f" % tnp1)

def vector_no_loop():
    import numpy as np
    import math
    from time import time
    np.random.seed(20000)
    t0 = time()
    
    # Parameters
    S0 = 100.0
    K=105.0
    T=1.0
    r=0.05
    sigma=0.2
    M = 50
    dt = T / M
    I = 250000
    # import pdb; pdb.set_trace()
    S = S0 * np.exp(np.cumsum((r - 0.5 * sigma **2) * dt + sigma * math.sqrt(dt) * np.random.standard_normal((M+1, I)), axis=0))
    S[0] = S0
    
    C0 = math.exp(-r * T) * sum(np.maximum(S[-1] - K, 0)) / I
    
    # Results output
    tnp1 = time() - t0
    print("European Option value %7.3f" % C0)
    print("Duration in Seconds   %7.3f" % tnp1)
    plt.plot(S[:, :10])
    plt.grid(True)
    plt.xlabel('time step')
    plt.ylabel('index level')
    plt.savefig('png/ch3/montecarlo.png', dpi=300)
    plt.close()
    
    plt.hist(S[-1], bins=50)
    plt.grid(True)
    plt.xlabel('index level')
    plt.ylabel('frequency')
    plt.savefig('png/ch3/mc_end_index_vals.png', dpi=300)
    plt.close()
    
    plt.hist(np.maximum(S[-1]-K,0), bins=50)
    plt.grid(True)
    plt.xlabel('option inner value')
    plt.ylabel('frequency')
    plt.ylim(0, 50000)
    plt.savefig('png/ch3/mc_option_end_value.png', dpi=300)
    plt.close()

def tech_analysis():
    import numpy as np
    import pandas as pd
    import pandas_datareader.data as web
    
    sp500 = web.DataReader('^GSPC', data_source='yahoo', start='1/1/2000', end='4/14/2014')
    sp500.info()
    sp500['42d'] = np.round(sp500['Close'].rolling(window=42, center=False).mean(), 2)
    sp500['252d'] = np.round(sp500['Close'].rolling(window=252, center=False).mean(), 2)
    # sp500['42d'] = np.round(pd.rolling_mean(sp500['Close'], window=42), 2)
    sp500['42-252'] = sp500['42d'] - sp500['252d']
    
    # Setting up the regime
    SD = 50
    sp500['Regime'] = np.where(sp500['42-252'] > SD, 1, 0)
    sp500['Regime'] = np.where(sp500['42-252'] < -SD, -1, sp500['Regime'])
    print(sp500['Regime'].value_counts())
    
    plt.plot(sp500['Close'])
    plt.plot(sp500['42d'])
    plt.plot(sp500['252d'])
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('index level')
    # sp500['Close'].plot(grid=True, figsize=(8,5))
    plt.savefig('png/ch3/sp500.png', dpi=300)
    plt.close()
    
    plt.plot(sp500['Regime'], lw=1.5)
    plt.ylim([-1.1, 1.1])
    plt.savefig('png/ch3/Regime.png', dpi=300)
    plt.close()
    
    sp500['Market'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
    sp500['Strategy'] = sp500['Regime'].shift(1) * sp500['Market']
    
    import pdb; pdb.set_trace()
    plt.plot(sp500[['Market','Strategy']].cumsum().apply(np.exp))
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('returns')
    plt.savefig('png/ch3/strat_vs_market.png', dpi=300)
    plt.close()
    

if __name__ == "__main__":
    # vector_no_loop()
    tech_analysis()