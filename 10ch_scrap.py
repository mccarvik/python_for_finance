import sys, pdb
sys.path.append('/usr/share/doc')
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as scs

PATH = '/home/ubuntu/workspace/python_for_finance/png/ch10/'

def randon_nums():
    print(npr.rand(10))
    print(npr.rand(5, 5))
    a = 5.
    b = 10.
    print(npr.rand(10) * (b - a) + a)
    print(npr.rand(5, 5) * (b - a) + a)
    sample_size = 500
    rn1 = npr.rand(sample_size, 3)
    rn2 = npr.randint(0, 10, sample_size)
    rn3 = npr.sample(size=sample_size)
    a = [0, 25, 50, 75, 100]
    rn4 = npr.choice(a, size=sample_size)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                                 figsize=(7, 7))
    ax1.hist(rn1, bins=25, stacked=True)
    ax1.set_title('rand')
    ax1.set_ylabel('frequency')
    ax1.grid(True)
    ax2.hist(rn2, bins=25)
    ax2.set_title('randint')
    ax2.grid(True)
    ax3.hist(rn3, bins=25)
    ax3.set_title('sample')
    ax3.set_ylabel('frequency')
    ax3.grid(True)
    ax4.hist(rn4, bins=25)
    ax4.set_title('choice')
    ax4.grid(True)
    plt.savefig(PATH + 'rand1.png', dpi=300)
    plt.close()

    sample_size = 500
    rn1 = npr.standard_normal(sample_size)
    rn2 = npr.normal(100, 20, sample_size)
    rn3 = npr.chisquare(df=0.5, size=sample_size)
    rn4 = npr.poisson(lam=1.0, size=sample_size)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    ax1.hist(rn1, bins=25)
    ax1.set_title('standard normal')
    ax1.set_ylabel('frequency')
    ax1.grid(True)
    ax2.hist(rn2, bins=25)
    ax2.set_title('normal(100, 20)')
    ax2.grid(True)
    ax3.hist(rn3, bins=25)
    ax3.set_title('chi square')
    ax3.set_ylabel('frequency')
    ax3.grid(True)
    ax4.hist(rn4, bins=25)
    ax4.set_title('Poisson')
    ax4.grid(True)
    plt.savefig(PATH + 'rand2.png', dpi=300)
    plt.close()

def rand_vals():
    S0 = 100  # initial value
    r = 0.05  # constant short rate
    sigma = 0.25  # constant volatility
    T = 2.0  # in years
    I = 10000  # number of random draws
    ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T 
                 + sigma * np.sqrt(T) * npr.standard_normal(I))
    plt.hist(ST1, bins=50)
    plt.xlabel('index level')
    plt.ylabel('frequency')
    plt.grid(True)
    plt.savefig(PATH + 'rand_vals1.png', dpi=300)
    plt.close()
    
    ST2 = S0 * npr.lognormal((r - 0.5 * sigma ** 2) * T,
                        sigma * np.sqrt(T), size=I)
    plt.hist(ST2, bins=50)
    plt.xlabel('index level')
    plt.ylabel('frequency')
    plt.grid(True)
    plt.savefig(PATH + 'rand_vals2.png', dpi=300)
    plt.close()
    print_statistics(ST1, ST2)
    
def print_statistics(a1, a2):
    ''' Prints selected statistics.
    
    Parameters
    ==========
    a1, a2 : ndarray objects
        results object from simulation
    '''
    sta1 = scs.describe(a1)
    sta2 = scs.describe(a2)
    print("%14s %14s %14s" % 
        ('statistic', 'data set 1', 'data set 2'))
    print(45 * "-")
    print("%14s %14.3f %14.3f" % ('size', sta1[0], sta2[0]))
    print("%14s %14.3f %14.3f" % ('min', sta1[1][0], sta2[1][0]))
    print("%14s %14.3f %14.3f" % ('max', sta1[1][1], sta2[1][1]))
    print("%14s %14.3f %14.3f" % ('mean', sta1[2], sta2[2]))
    print("%14s %14.3f %14.3f" % ('std', np.sqrt(sta1[3]), np.sqrt(sta2[3])))
    print("%14s %14.3f %14.3f" % ('skew', sta1[4], sta2[4]))
    print("%14s %14.3f %14.3f" % ('kurtosis', sta1[5], sta2[5]))
    
def stochastic_procs():
    I = 10000
    M = 50
    T = 2.0  # in years
    S0 = 100  # initial value
    r = 0.05  # constant short rate
    sigma = 0.25  # constant volatility
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T 
                 + sigma * np.sqrt(T) * npr.standard_normal(I))
    ST2 = S0 * npr.lognormal((r - 0.5 * sigma ** 2) * T,
                        sigma * np.sqrt(T), size=I)
    
    # Geometric Brownian motion
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt 
                + sigma * np.sqrt(dt) * npr.standard_normal(I))
    plt.hist(S[-1], bins=50)
    plt.xlabel('index level')
    plt.ylabel('frequency')
    plt.grid(True)
    plt.savefig(PATH + 'stoch_procs1.png', dpi=300)
    plt.close()

    print_statistics(S[-1], ST2)
    plt.plot(S[:, :10], lw=1.5)
    plt.xlabel('time')
    plt.ylabel('index level')
    plt.grid(True)
    plt.savefig(PATH + 'stoch_procs2.png', dpi=300)
    plt.close()

def sqr_rt_diffusion():
    pass

if __name__ == '__main__':
    # randon_nums()
    # rand_vals()
    stochastic_procs()