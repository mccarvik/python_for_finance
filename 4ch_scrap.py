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
print(type(a))