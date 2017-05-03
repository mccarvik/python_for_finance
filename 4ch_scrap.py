# combining objects works with reference pointers, example:
v = [5, 75, 1]
m = [v, v, v,]
# print(m)
v[0] = 'python'
# print(m)

# to avoid this use deep copy, example:
from copy import deepcopy
v = [5, 75, 1]
m = 3 * [deepcopy(v), ]
# print(m)
v[0]= 'python'
# print(m)

# ndarray
import numpy as np
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
import random, time
from functools import reduce
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