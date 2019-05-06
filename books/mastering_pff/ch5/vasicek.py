""" Simulate interest rate path by the Vasicek model """
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

IMG_PATH = "/home/ubuntu/workspace/python_for_finance/mastering_pff/png/"

def vasicek(r0, K, theta, sigma, T=1., N=10, seed=777):    
    np.random.seed(seed)
    dt = T/float(N)    
    rates = [r0]
    for i in range(N):
        # theta = mean rate, kappa = mean reversion coeff, sigma = stdev, rand = wiener process rand val
        # takes the previous rate as a starting point
        # no drift, just bounces around the mean
        # Can be negative
        dr = K*(theta-rates[-1])*dt + sigma*np.random.normal()
        rates.append(rates[-1] + dr)
    return range(N+1), rates

if __name__ == "__main__":
    x, y = vasicek(0.01875, 0.20, 0.01, 0.012, 10., 200)
    plt.plot(x,y)
    plt.savefig(IMG_PATH + 'vasicek.png', dpi=300)
    plt.close()