import random, time, pdb, math
import numpy as np
from functools import reduce
from copy import deepcopy

def intro():
    # combining objects works with reference pointers, example:
    v = [5, 75, 1]
    m = [v, v, v,]
    # print(m)
    v[0] = 'python'
    # print(m)
    
    # to avoid this use deep copy, example:
    v = [5, 75, 1]
    m = 3 * [deepcopy(v), ]
    # print(m)
    v[0]= 'python'
    # print(m)
    
    # ndarray
    a = np.array([0, 0.5, 1.0, 1.5, 2.0])
    # print(type(a))
    # print(a[:2]) # slicing like list objects
    # print(a.sum(), a.std(), a.cumsum())     # built in array functions
    # print(a*2, a**2, np.sqrt(a))            # vectorized functions
    
    # multi-dimensional
    b = np.array([a, a * 2])
    # print(b)
    # print(b[0], b.sum(), b.sum(axis=0), b.sum(axis=1))
    
    c = np.zeros((2,3,4), dtype='i', order='C')
    # print(c)
    
    d = np.ones_like(c, dtype='f16', order='C')
    # print(d)
    
    # random
    I = 5000
    t0 = time.time()
    mat = [[random.gauss(0,1) for j in range(I) for i in range(I)]]
    t1 = time.time()
    print("took {0} seconds".format(t1-t0))
    
    t0 = time.time()
    sm = reduce(lambda x, y: x+y, [reduce(lambda x, y: x+y, row) for row in mat])
    t1 = time.time()
    print("took {0} seconds".format(t1-t0))
    print(sm)
    
    t0 = time.time()
    mat = np.random.standard_normal((I, I))
    print(mat.sum())
    t1 = time.time()
    print("took {0} seconds".format(t1-t0))

def structured_arr():
    dt = np.dtype([('Name', 'S10'), ('Age', 'i4'), 
                    ('Height', 'f'), ('Children/Pets', 'i4', 2)])
    s = np.array([('Smith', 45, 1.83, (0,1)), 
                ('Jones', 53, 1.72, (2, 2))], dtype=dt)
    print(s)

def vectorization():
    r = np.random.standard_normal((4, 3))
    s = np.random.standard_normal((4, 3))
    print(r + s)
    print(2 * r + 3)
    
    s = np.random.standard_normal(3)
    print(r+s)
    s = np.random.standard_normal(4)
    # print(r+s)        # causes error as arrays not same compatible shapes
    print(r.transpose()+s)
    print(np.shape(r.T))
    print(f(0.5))
    print(f(r))
    # math.sin(r)         # causes a bug, fnction not set up to handle arrays
    print(np.sin(r))
    print(np.sin(np.pi))
    
def memory():
    x = np.random.standard_normal((5, 1000000))
    y = 2 * x + 3
    C = np.array((x, y), order='C')
    F = np.array((x, y), order='F')
    x = 0.0; y = 0.0    # memory cleanup
    # print(C[:2].round(2))
    
    t0 = time.time()
    C.sum()
    t1 = time.time()
    print("took {0} seconds".format(t1-t0))
    
    t0 = time.time()
    F.sum()
    t1 = time.time()
    print("took {0} seconds".format(t1-t0))
    
    t0 = time.time()
    C[0].sum(axis=0)
    t1 = time.time()
    print("took {0} seconds".format(t1-t0))
    
    t0 = time.time()
    C[0].sum(axis=1)
    t1 = time.time()
    print("took {0} seconds".format(t1-t0))
    
    t0 = time.time()
    F.sum(axis=0)
    t1 = time.time()
    print("took {0} seconds".format(t1-t0))
    
    t0 = time.time()
    F.sum(axis=1)
    t1 = time.time()
    print("took {0} seconds".format(t1-t0))
    

def f(x):
    return 3 * x + 5

if __name__ ==  "__main__":
    # pdb.set_trace()
    # structured_arr()
    # vectorization()
    memory()
    
    
    
    
    
    
    
    
    