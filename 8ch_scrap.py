import sys
sys.path.append("/usr/local/lib/python3.4/dist-packages")
sys.path.append("/usr/local/lib/python3.5/dist_packages")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb, time, timeit
from math import *
import numexpr as ne
import numpy as np
# import numba as nb
import multiprocessing as mp


PATH = '/home/ubuntu/workspace/python_for_finance/png/ch8/'

I = 100
J = 100
K = 100.
M = 20  # number of time steps
t = 10  # number of tasks/simulations
a_py = range(I)
a_np = np.arange(I)
n = 10  # number of option valuations

def perf_comp_data(func_list, data_list, rep=3, number=1):
    ''' Function to compare the performance of different functions.
    
    Parameters
    ==========
    func_list : list
        list with function names as strings
    data_list : list
        list with data set names as strings
    rep : int
        number of repetitions of the whole comparison
    number : int
        number of executions for every function
    '''
    
    
    from timeit import repeat
    res_list = {}
    for name in enumerate(func_list):
        stmt = name[1] + '(' + data_list[name[0]] + ')'
        setup = "from __main__ import " + name[1] + ', ' \
                                    + data_list[name[0]]
        results = repeat(stmt=stmt, setup=setup,
                         repeat=rep, number=number)
        res_list[name[1]] = sum(results) / rep
    res_sort = sorted(res_list.items(),
                      key=lambda x: (x[1], x[0]))
    for item in res_sort:
        rel = item[1] / res_sort[0][1]
        print ('function: ' + item[0] +
              ', av. time sec: %9.5f, ' % item[1]
            + 'relative: %6.1f' % rel)

def f(x):
    return abs(cos(x)) ** 0.5 + sin(2 + 3 * x)

def f1(a):
    res = []
    for x in a:
        res.append(f(x))
    return res

def f2(a):
    return [f(x) for x in a]

def f3(a):
    ex = 'abs(cos(x)) ** 0.5 + sin(2 + 3 * x)'
    return [eval(ex) for x in a]

def f4(a):
    return (np.abs(np.cos(a)) ** 0.5 +
            np.sin(2 + 3 * a))

def f5(a):
    ex = 'abs(cos(a)) ** 0.5 + sin(2 + 3 * a)'
    ne.set_num_threads(1)
    return ne.evaluate(ex)

def f6(a):
    ex = 'abs(cos(a)) ** 0.5 + sin(2 + 3 * a)'
    ne.set_num_threads(16)
    return ne.evaluate(ex)

def paradigms():
    r1 = f1(a_py)
    r2 = f2(a_py)
    r3 = f3(a_py)
    r4 = f4(a_np)
    r5 = f5(a_np)
    r6 = f6(a_np)
    np.allclose(r1, r2)
    np.allclose(r1, r3)
    np.allclose(r1, r4)
    np.allclose(r1, r5)
    np.allclose(r1, r6)
    func_list = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
    data_list = ['a_py', 'a_py', 'a_py', 'a_np', 'a_np', 'a_np']
    perf_comp_data(func_list, data_list)

def memory_layout():
    print(np.zeros((3, 3), dtype=np.float64, order='C'))
    c = np.array([[ 1.,  1.,  1.],
                  [ 2.,  2.,  2.],
                  [ 3.,  3.,  3.]], order='C')
    f = np.array([[ 1.,  1.,  1.],
                  [ 2.,  2.,  2.],
                  [ 3.,  3.,  3.]], order='F')
    x = np.random.standard_normal((3, 150000))
    C = np.array(x, order='C')
    F = np.array(x, order='F')
    x = 0.0
    timeme(C.sum)(axis=0)
    timeme(C.sum)(axis=1)
    timeme(C.std)(axis=0)
    timeme(C.std)(axis=1)
    timeme(F.sum)(axis=0)
    timeme(F.sum)(axis=1)
    timeme(F.std)(axis=0)
    timeme(F.std)(axis=1)
    C = 0.0; F = 0.0
    
def bsm_mcs_valuation(strike):
    ''' Dynamic Black-Scholes-Merton Monte Carlo estimator
    for European calls.
    
    Parameters
    ==========
    strike : float
        strike price of the option
    
    Results
    =======
    value : float
        estimate for present value of call option
    '''
    S0 = 100.; T = 1.0; r = 0.05; vola = 0.2
    M = 50; I = 20000
    dt = T / M
    rand = np.random.standard_normal((M + 1, I))
    S = np.zeros((M + 1, I)); S[0] = S0
    for t in range(1, M + 1):
        S[t] = S[t-1] * np.exp((r - 0.5 * vola ** 2) * dt
                               + vola * np.sqrt(dt) * rand[t])
    value = (np.exp(-r * T)
                     * np.sum(np.maximum(S[-1] - strike, 0)) / I)
    return value
    
def simulate_geometric_brownian_motion(p):
    M, I = p
      # time steps, paths
    S0 = 100; r = 0.05; sigma = 0.2; T = 1.0
      # model parameters
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                    sigma * sqrt(dt) * np.random.standard_normal(I))
    return paths

def seq_value(n):
    ''' Sequential option valuation.
    
    Parameters
    ==========
    n : int
        number of option valuations/strikes
    '''
    strikes = np.linspace(80, 120, n)
    option_values = []
    for strike in strikes:
        option_values.append(bsm_mcs_valuation(strike))
    return strikes, option_values

def parallel_analysis(num):
    ''' Parallel option valuation.
    
    Parameters
    ==========
    n : int
        number of option valuations/strikes
    '''
    strikes = np.linspace(80, 120, num)
    from time import time
    times = []
    for w in range(1, 17):
        t0 = time()
        pool = mp.Pool(processes=w)
        # the pool of workers
        # result = pool.map(bsm_mcs_valuation, strikes)
        result = pool.map(simulate_geometric_brownian_motion, t * [(M, I), ])
        # the mapping of the function to the list of parameter tuples
        times.append(time() - t0)
    plt.plot(range(1, 17), times)
    plt.plot(range(1, 17), times, 'ro')
    plt.grid(True)
    plt.xlabel('number of processes')
    plt.ylabel('time in seconds')
    plt.title('Monte Carlo simulations')
    plt.savefig(PATH + 'multiprocessing.png', dpi=300)
    plt.close()

def par_value(n):
    ''' Parallel option valuation.
    
    Parameters
    ==========
    n : int
        number of option valuations/strikes
    '''
    strikes = np.linspace(80, 120, n)
    option_values = {}
    pool = mp.Pool(processes = 10)
    option_values = pool.map(bsm_mcs_valuation, strikes)
    return strikes, option_values
    
def parallel_computing():
    n = 10  # number of options to be valued
    strikes, option_values_seq = timeme(seq_value)(n)
    plt.figure(figsize=(8, 4))
    plt.plot(strikes, option_values_seq, 'b')
    plt.plot(strikes, option_values_seq, 'r.')
    plt.grid(True)
    plt.xlabel('strikes')
    plt.ylabel('European call option values')
    plt.savefig(PATH + 'para_computing.png', dpi=300)
    plt.close()
    
    strikes, option_values_par = timeme(par_value)(n)
    print(option_values_par[0])
    plt.figure(figsize=(8, 4))
    plt.plot(strikes, option_values_seq, 'b', label='Sequential')
    plt.plot(strikes, option_values_par, 'r.', label='Parallel')
    plt.grid(True); plt.legend(loc=0)
    plt.xlabel('strikes')
    plt.ylabel('European call option values')
    plt.savefig(PATH + 'para_computing2.png', dpi=300)
    plt.close()
    
    func_list = ['seq_value', 'par_value']
    data_list = 2 * ['n']
    perf_comp_data(func_list, data_list)
    
def timeme(method):
    def wrapper(*args, **kw):
        startTime = round(time.time() * 1000, 3)
        result = method(*args, **kw)
        endTime = round(time.time() * 1000, 3)
        print(endTime - startTime,'ms')
        return result
    return wrapper

def f_py(I, J):
    res = 0
    for i in range(I):
        for j in range (J):
            res += int(cos(log(1)))
    return res

def f_np(I, J):
    a = np.ones((I, J), dtype=np.float64)
    return int(np.sum(np.cos(np.log(a)))), a
    
def dynamic_compiling():
    timeme(f_py)(I, J)
    res, a = timeme(f_np)(I, J)
    print(a.nbytes)
    
    pdb.set_trace()
    f_nb = nb.jit(f_py)
    timeme(f_nb)(I, J)
    func_list = ['f_py', 'f_np', 'f_nb']
    data_list = 3 * ['I, J']
    perf_comp_data(func_list, data_list)
    
    print(timeme(binomial_py_looping)(100))
    print(timeme(bsm_mcs_valuation)(100))
    print(timeme(binomial_np)(100))
    
    binomial_nb = nb.jit(binomial_py)
    print(timeme(binomial_nb)(100))
    func_list = ['binomial_py', 'binomial_np', 'binomial_nb']
    data_list = 3 * ['K']
    perf_comp_data(func_list, data_list)

def binomial_py_looping(strike):
    ''' Binomial option pricing via looping.
    
    Parameters
    ==========
    strike : float
        strike price of the European call option
    '''
    
    S0 = 100.  # initial index level
    T = 1.  # call option maturity
    r = 0.05  # constant short rate
    vola = 0.20  # constant volatility factor of diffusion
    
    # time parameters
    M = 1000  # time steps
    dt = T / M  # length of time interval
    df = exp(-r * dt)  # discount factor per time interval
    
    # binomial parameters
    u = exp(vola * sqrt(dt))  # up-movement
    d = 1 / u  # down-movement
    q = (exp(r * dt) - d) / (u - d)  # martingale probability
    
    
    # LOOP 1 - Index Levels
    S = np.zeros((M + 1, M + 1), dtype=np.float64)
      # index level array
    S[0, 0] = S0
    z1 = 0
    for j in range(1, M + 1, 1):
        z1 = z1 + 1
        for i in range(z1 + 1):
            S[i, j] = S[0, 0] * (u ** j) * (d ** (i * 2))
            
    # LOOP 2 - Inner Values
    iv = np.zeros((M + 1, M + 1), dtype=np.float64)
      # inner value array
    z2 = 0
    for j in range(0, M + 1, 1):
        for i in range(z2 + 1):
            iv[i, j] = max(S[i, j] - strike, 0)
        z2 = z2 + 1
        
    # LOOP 3 - Valuation
    pv = np.zeros((M + 1, M + 1), dtype=np.float64)
      # present value array
    pv[:, M] = iv[:, M]  # initialize last time point
    z3 = M + 1
    for j in range(M - 1, -1, -1):
        z3 = z3 - 1
        for i in range(z3):
            pv[i, j] = (q * pv[i, j + 1] +
                        (1 - q) * pv[i + 1, j + 1]) * df
    return pv[0, 0]
    
def binomial_np(strike):
    ''' Binomial option pricing with NumPy.
    
    Parameters
    ==========
    strike : float
        strike price of the European call option
    '''
    S0 = 100.  # initial index level
    T = 1.  # call option maturity
    r = 0.05  # constant short rate
    vola = 0.20  # constant volatility factor of diffusion
    
    # time parameters
    M = 1000  # time steps
    dt = T / M  # length of time interval
    df = exp(-r * dt)  # discount factor per time interval
    
    # binomial parameters
    u = exp(vola * sqrt(dt))  # up-movement
    d = 1 / u  # down-movement
    q = (exp(r * dt) - d) / (u - d)  # martingale probability
    
    
    # Index Levels with NumPy
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md
    
    # Valuation Loop
    pv = np.maximum(S - strike, 0)

    z = 0
    for t in range(M - 1, -1, -1):  # backwards iteration
        pv[0:M - z, t] = (q * pv[0:M - z, t + 1]
                        + (1 - q) * pv[1:M - z + 1, t + 1]) * df
        z += 1
    return pv[0, 0]

def static_comp_cython():
    print(timeme(f_py)(I, J))

    import pyximport
    pyximport.install()
    import sys
    sys.path.append('data/')
    # path to the Cython script
    # not needed if in same directory
    from nested_loop import f_cy
    
    print(timeme(f_cy)(I, J))
    # %load_ext Cython
    import numba as nb
    f_nb = nb.jit(f_py)
    print(timeme(f_nb)(I, J))
    func_list = ['f_py', 'f_cy', 'f_nb']
    data_list = 3 * ['I, J']
    perf_comp_data(func_list, data_list)

def f_py(I, J):
    res = 0.  # we work on a float object
    for i in range(I):
        for j in range (J * I):
            res += 1
    return res

# def f_cy(int I, int J):
#     cdef double res = 0
#     # double float much slower than int or long
#     for i in range(I):
#         for j in range (J * I):
#             res += 1
#     return res

if __name__ == '__main__':
    # paradigms()
    # memory_layout()
    parallel_computing()
    parallel_analysis(5)
    # dynamic_compiling()