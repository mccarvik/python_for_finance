from bsm_functions import bsm_call_value

# Parameters
S0 = 100.0
K=105.0
T=1.0
r=0.05
sigma=0.2
print(bsm_call_value(S0, K, T, r, sigma))

from time import time
from math import exp, sqrt, log
from random import gauss, seed

seed(20000)
t0 = time()

M = 50
dt = T / M
I = 250000

# Simulating I paths with M time steps
S = []
for i in range(I):
    path = []
    for t in range(M+1):
        if t==0:
            path.append(S0)
        else:
            z = gauss(0.0, 1.0)
            St = path[t-1] * exp((r-0,5 * sigma ** 2) * dt + sigma * sqrt(dt)*z)
            path.append(St)
    S.append(path)

# Calculating the Monte Carlo Estimator
C0 = exp(-r * T) * sum([max(path[-1] - K, 0) for path in S]) / I

# Results output
tpy = time() - t0
print("European OPtion value %7.3f" % C0)
print("Duration in Seconds   %7.3f" % tpy)