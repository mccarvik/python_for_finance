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
import multiprocessing as mp
import numba as nb

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

def multiprocessing():
    paths = simulate_geometric_brownian_motion((5, 2))
    print(paths)
    I = 1000  # number of paths
    M = 50  # number of time steps
    t = 20  # number of tasks/simulations
    times = []
    for w in range(1, 5):
        t0 = time.time()
        pool = mp.Pool(processes=w)
        # the pool of workers
        result = pool.map(simulate_geometric_brownian_motion, t * [(M, I), ])
        # the mapping of the function to the list of parameter tuples
        times.append(time.time() - t0)
    plt.plot(range(1, 5), times)
    plt.plot(range(1, 5), times, 'ro')
    plt.grid(True)
    plt.xlabel('number of processes')
    plt.ylabel('time in seconds')
    plt.title('%d Monte Carlo simulations' % t)
    plt.savefig(PATH + 'parallel.png', dpi=300)
    plt.close()
    
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

def dynamic_compiling():
    I, J = 2500, 2500
    %time f_py(I, J)
    %time res, a = f_np(I, J)
    a.nbytes
    f_nb = nb.jit(f_py)
    %time f_nb(I, J)
    func_list = ['f_py', 'f_np', 'f_nb']
    data_list = 3 * ['I, J']
    perf_comp_data(func_list, data_list)

def f_py(I, J):
    res = 0
    for i in range(I):
        for j in range (J):
            res += int(cos(log(1)))
    return res
    
def f_np(I, J):
    a = np.ones((I, J), dtype=np.float64)
    return int(np.sum(np.cos(np.log(a)))), a


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
    # parallel_computing()
    multiprocessing()