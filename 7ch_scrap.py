import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb, requests, math, requests, datetime, pickle
import numpy as np
import pandas as pd
from random import gauss

path = '/home/ubuntu/workspace/python_for_finance/data'

def pkl():
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

def read_write():
    rows = 5000
    a = np.random.standard_normal((rows, 5))  # dummy data
    t = pd.date_range(start='2014/1/1', periods=rows, freq='H')
    csv_file = open(path + 'data.csv', 'w')  # open file for writing
    header = 'date,no1,no2,no3,no4,no5\n'
    csv_file.write(header)
    for t_, (no1, no2, no3, no4, no5) in zip(t, a):
        s = '%s,%f,%f,%f,%f,%f\n' % (t_, no1, no2, no3, no4, no5)
        csv_file.write(s)
    csv_file.close()
    
    csv_file = open(path + 'data.csv', 'r')  # open file for reading
    for i in range(5):
        print(csv_file.readline(), end='')
    
    csv_file = open(path + 'data.csv', 'r')
    content = csv_file.readlines()
    for line in content[:5]:
        print(line, end='')
        
    csv_file.close()
    
def sql_db():
    pass
        
    
    
    
    
    
    


if __name__ == "__main__":
    # pkl()
    read_write()