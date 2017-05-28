import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb, requests, math, requests, datetime, pickle
import numpy as np
import pandas as pd
from random import gauss

def pkl():
    pdb.set_trace()
    path = '/home/ubuntu/workspace/python_for_finance/data'
    a = [gauss(1.5, 2) for i in range(10000)]
    pkl_file = open(path + 'data.pkl', 'wb')
    pickle.dump(a, pkl_file)
    pkl_file.close()
    
    pkl_file = open(path + 'data.pkl', 'rb')
    b = pickle.load(pkl_file)
    print(np.allclose(np.array(a), np.array(b)))
    
    pkl_file = open(path + 'data.pkl', 'wb')
    pickle.dump(np.array(a), pkl_file)
    pickle.dump(np.array(a) ** 2, pkl_file)
    pkl_file.close()
    pkl_file = open(path + 'data.pkl', 'rb')
    x = pickle.load(pkl_file)
    y = pickle.load(pkl_file)
    print(x)
    print(y)
    pkl_file.close()
    
    pkl_file = open(path + 'data.pkl', 'wb')
    pickle.dump({'x' : x, 'y': y}, pkl_file)
    pkl_file.close()
    
    pkl_file = open(path + 'data.pkl', 'rb')  # open file for writing
    data = pickle.load(pkl_file)
    pkl_file.close()
    for key in data.keys():
        print(key, data[key][:4])
    
    
    
    


if __name__ == "__main__":
    pkl()