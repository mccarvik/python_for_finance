#
# Valuation of American Options
# with Least-Squares Monte Carlo
# Primal Algorithm
# American Put Option
# 07_amo/LSM_primal_valuation.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np
np.random.seed(150000)

# Model Parameters
S0 = 36.  # initial stock level
K = 40.  # strike price
T = 1.0  # time-to-maturity
r = 0.06  # short rate
sigma = 0.2  # volatility

# Simulation Parameters
I = 2500
M = 50
dt = T / M
df = math.exp(-r * dt)

# Stock Price Paths
S = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt +
                          sigma * math.sqrt(dt) *
                          np.random.standard_normal((M + 1, I)), axis=0))
S[0] = S0

# Inner Values
h = np.maximum(K - S, 0)

# Present Value Vector (Initialization)
V = h[-1]

# American Option Valuation by Backwards Induction
# step backward from end state
for t in range(M - 1, 0, -1):
    # Uses regression to deduce the continuation values
    rg = np.polyfit(S[t], V * df, 5)
    C = np.polyval(rg, S[t])  # continuation values
    # V = to the max of exercising or not
    V = np.where(h[t] > C, h[t], V * df)
    # exercise decision
V0 = df * np.sum(V) / I  # LSM estimator

print("American put option value %5.3f" % V0)