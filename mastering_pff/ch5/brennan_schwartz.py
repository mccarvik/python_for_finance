""" Simulate interest rate path by the Brennan Schwartz model """
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

IMG_PATH = "/home/ubuntu/workspace/python_for_finance/mastering_pff/png/"

def brennan_schwartz(r0, K, theta, sigma, T=1., N=10, seed=777):    
    np.random.seed(seed)
    dt = T/float(N)    
    rates = [r0]
    for i in range(N):
        dr = K*(theta-rates[-1])*dt + \
            sigma*rates[-1]*np.random.normal()
        rates.append(rates[-1] + dr)
    return range(N+1), rates

if __name__ == "__main__":
    x, y = brennan_schwartz(0.01875, 0.20, 0.01, 0.012, 10., 10000)
    plt.plot(x,y)
    plt.savefig(IMG_PATH + 'brennan_schwartz.png', dpi=300)
    plt.close()