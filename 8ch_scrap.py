import sys
sys.path.append("/usr/local/lib/python3.5")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb, time, timeit
from math import *
import numexpr as ne
import numpy as np

pdb.set_trace()
from ipyparallel import Client
c = Client(profile="default")
view = c.load_balanced_view()

PATH = '/home/ubuntu/workspace/python_for_finance/png/ch8/'

I = 50000
a_py = range(I)
a_np = np.arange(I)

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
    import numpy as np
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

def parallel_computing():
    n = 10  # number of options to be valued
    strikes, option_values_seq = timeme(seq_value)(n)
    plt.figure(figsize=(8, 4))
    plt.plot(strikes, option_values_seq, 'b')
    plt.plot(strikes, option_values_seq, 'r.')
    plt.grid(True)
    plt.xlabel('strikes')
    plt.ylabel('European call option values')
    plt.savefig(PATH + 'options.png', dpi=300)
    plt.close()
    
    strikes, option_values_seq = timeme(par_value)(n)
    option_values_obj[0].metadata
    option_values_par = []
    for res in option_values_obj:
        option_values_par.append(res.result)
    plt.figure(figsize=(8, 4))
    plt.plot(strikes, option_values_seq, 'b', label='Sequential')
    plt.plot(strikes, option_values_par, 'r.', label='Parallel')
    plt.grid(True); plt.legend(loc=0)
    plt.xlabel('strikes')
    plt.ylabel('European call option values')
    plt.savefig(PATH + 'parallel.png', dpi=300)
    plt.close()

def par_value(n):
    ''' Parallel option valuation.
    
    Parameters
    ==========
    n : int
        number of option valuations/strikes
    '''
    strikes = np.linspace(80, 120, n)
    option_values = []
    for strike in strikes:
        value = view.apply_async(bsm_mcs_valuation, strike)
        option_values.append(value)
    c.wait(option_values)
    return strikes, option_values

def timeme(method):
    def wrapper(*args, **kw):
        startTime = round(time.time() * 1000, 3)
        result = method(*args, **kw)
        endTime = round(time.time() * 1000, 3)
        print(endTime - startTime,'ms')
        return result
    return wrapper

if __name__ == '__main__':
    # paradigms()
    # memory_layout()
    parallel_computing()