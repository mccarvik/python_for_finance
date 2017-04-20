from bsm_functions import bsm_call_value

def pure_python():
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
    import pdb; pdb.set_trace()
    for i in range(I):
        path = []
        for t in range(M+1):
            if t==0:
                path.append(S0)
            else:
                z = gauss(0.0, 1.0)
                St = path[t-1] * exp((r-0.5 * sigma ** 2) * dt + sigma * sqrt(dt)*z)
                path.append(St)
        S.append(path)
    
    # Calculating the Monte Carlo Estimator
    C0 = exp(-r * T) * sum([max(path[-1] - K, 0) for path in S]) / I
    
    # Results output
    tpy = time() - t0
    print("European Option value %7.3f" % C0)
    print("Duration in Seconds   %7.3f" % tpy)

def numpy_vectors():
    import numpy as np
    import math
    from time import time
    np.random.seed(20000)
    t0 = time()
    
    # Parameters
    S0 = 100.0
    K=105.0
    T=1.0
    r=0.05
    sigma=0.2
    M = 50
    dt = T / M
    I = 250000
    
    # Simulating I paths with M time steps
    S = np.zeros((M+1, I))
    S[0] = S0
    import pdb; pdb.set_trace()
    for t in range(1, M + 1):
        z = np.random.standard_normal(I) # pseudorandom numbers
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z)
        # vectorized operation per time step over all paths
    
    C0 = math.exp(-r * T) * np.sum(np.maximum(S[-1] - K, 0)) / I
    
    # Results output
    tnp1 = time() - t0
    print("European Option value %7.3f" % C0)
    print("Duration in Seconds   %7.3f" % tnp1)

def vector_no_loop():
    import numpy as np
    import math
    from time import time
    np.random.seed(20000)
    t0 = time()
    
    # Parameters
    S0 = 100.0
    K=105.0
    T=1.0
    r=0.05
    sigma=0.2
    M = 50
    dt = T / M
    I = 250000
    
    S = S0 * math.exp(np.cumsum((r - 0.5 * sigma **2) * dt + sigma * math.sqrt(dt) * \
                        np.random.standard_normal((M+1, I)), axis=0))
    S[0] = S0
    
    C0 = math.exp(-r * T) * sum(maximum(S[-1] - K, 0)) / I
    
    # Results output
    tnp1 = time() - t0
    print("European Option value %7.3f" % C0)
    print("Duration in Seconds   %7.3f" % tnp1)
    
vector_no_loop()