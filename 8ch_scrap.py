import pdb
from math import *
import numexpr as ne
import numpy as np
import timeit

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
    pdb.set_trace()
    # print(timeit.Timer("C.sum(axis=0)").timeit(number=1))
    # print(timeit.Timer("C.sum(axis=1)").timeit(number=1))
    # print(timeit.Timer("C.std(axis=0)").timeit(number=1))
    # print(timeit.Timer("C.std(axis=1)").timeit(number=1))
    # print(timeit.Timer("F.sum(axis=0)").timeit(number=1))
    # print(timeit.Timer("F.sum(axis=1)").timeit(number=1))
    # print(timeit.Timer("F.std(axis=0)").timeit(number=1))
    # print(timeit.Timer("F.std(axis=1)").timeit(number=1))
    C = 0.0; F = 0.0

if __name__ == '__main__':
    # paradigms()
    memory_layout()