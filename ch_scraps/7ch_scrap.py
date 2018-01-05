import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb, requests, math, requests, datetime, pickle
import numpy as np
import pandas as pd
import sqlite3 as sq3
import datetime as dt
import tables as tb
import pandas.io.sql as pds
from random import gauss

path = '/home/ubuntu/workspace/python_for_finance/data'
PNG_PATH = '/home/ubuntu/workspace/python_for_finance/png/book_examples/ch7/'

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
    query = 'CREATE TABLE numbs (Date date, No1 real, No2 real)'
    con = sq3.connect(path + 'numbs.db')
    con.execute(query)
    con.commit()
    con.execute('INSERT INTO numbs VALUES(?, ?, ?)',
            (dt.datetime.now(), 0.12, 7.3))
    data = np.random.standard_normal((10000, 2)).round(5)
    for row in data:
        con.execute('INSERT INTO numbs VALUES(?, ?, ?)',
                    (dt.datetime.now(), row[0], row[1]))
    con.commit()
    con.execute('SELECT * FROM numbs').fetchmany(10)
    pointer = con.execute('SELECT * FROM numbs')
    for i in range(3):
        print(pointer.fetchone())    
    con.close()
        
def write_read_np_arrs():
    dtimes = np.arange('2015-01-01 10:00:00', '2021-12-31 22:00:00',
                      dtype='datetime64[m]')  # minute intervals
    print(len(dtimes))
    dty = np.dtype([('Date', 'datetime64[m]'), ('No1', 'f'), ('No2', 'f')])
    data = np.zeros(len(dtimes), dtype=dty)
    data['Date'] = dtimes
    a = np.random.standard_normal((len(dtimes), 2)).round(5)
    data['No1'] = a[:, 0]
    data['No2'] = a[:, 1]
    np.save(path + 'array', data)  # suffix .npy is added
    print(np.load(path + 'array.npy'))
    
def pandas_io():
    data = np.random.standard_normal((1000, 5)).round(5)
    # sample data set
    filename = path + 'numbs'
    # query = 'CREATE TABLE numbers (No1 real, No2 real,\
    #     No3 real, No4 real, No5 real)'
    con = sq3.Connection(filename + '.db')
    
    # Dont want to do these every run
    # con.execute(query)
    # con.executemany('INSERT INTO numbers VALUES (?, ?, ?, ?, ?)', data)
    # con.commit()
    temp = con.execute('SELECT * FROM numbers').fetchall()
    print(temp[:2])
    
    query = 'SELECT * FROM numbers WHERE No1 > 0 AND No2 < 0'
    res = np.array(con.execute(query).fetchall()).round(3)
    plt.plot(res[:, 0], res[:, 1], 'ro')
    plt.grid(True); plt.xlim(-0.5, 4.5); plt.ylim(-4.5, 0.5)
    plt.savefig(PNG_PATH + 'query.png', dpi=300)
    plt.close()
    
    data = pds.read_sql('SELECT * FROM numbers', con)
    print(data.head())
    print(data[(data['No1'] > 0) & (data['No2'] < 0)].head())
    
    res = data[['No1', 'No2']][((data['No1'] > 0.5) | (data['No1'] < -0.5))
                     & ((data['No2'] < -1) | (data['No2'] > 1))]
    plt.plot(res.No1, res.No2, 'ro')
    plt.grid(True); plt.axis('tight')
    plt.savefig(PNG_PATH + 'x_scatter.png', dpi=300)
    plt.close()
    
    h5s = pd.HDFStore(filename + '.h5s', 'w')
    h5s['data'] = data
    print(h5s)
    h5s.close()
    
    h5s = pd.HDFStore(filename + '.h5s', 'r')
    temp = h5s['data']
    h5s.close()
    
    np.allclose(np.array(temp), np.array(data))
    
    data.to_csv(filename + '.csv')
    # REMEMBER THISSSSSSSSSSSSSSSSSSS
    # Can do mpl on pandas or numpy and then just call plot() and savefig
    a = pd.read_csv(filename + '.csv')[['No1', 'No2',
                                'No3', 'No4']].hist(bins=20)
    plt.plot()
    plt.savefig(PNG_PATH + 'hist.png', dpi=300)
    plt.close()
    
    data[:1000].to_excel(filename + '.xlsx')
    pd.read_excel(filename + '.xlsx', 'Sheet1').cumsum().plot()
    plt.plot()
    plt.savefig(PNG_PATH + 'excel.png', dpi=300)
    plt.close()
    
    filename = path + 'tab.h5'
    h5 = tb.open_file(filename, 'w')
    rows = 1000
    row_des = {
    'Date': tb.StringCol(26, pos=1),
    'No1': tb.IntCol(pos=2),
    'No2': tb.IntCol(pos=3),
    'No3': tb.Float64Col(pos=4),
    'No4': tb.Float64Col(pos=5)
    }
    filters = tb.Filters(complevel=0)  # no compression
    tab = h5.create_table('/', 'ints_floats', row_des,
                      title='Integers and Floats',
                      expectedrows=rows, filters=filters)
    print(tab)
    
    pointer = tab.row
    ran_int = np.random.randint(0, 10000, size=(rows, 2))
    ran_flo = np.random.standard_normal((rows, 2)).round(5)
    for i in range(rows):
        pointer['Date'] = dt.datetime.now()
        pointer['No1'] = ran_int[i, 0]
        pointer['No2'] = ran_int[i, 1] 
        pointer['No3'] = ran_flo[i, 0]
        pointer['No4'] = ran_flo[i, 1] 
        pointer.append()
        # this appends the data and
        # moves the pointer one row forward
    tab.flush()  # flush = commit in sqlite
    print(tab)
    
    
    dty = np.dtype([('Date', 'S26'), ('No1', '<i4'), ('No2', '<i4'),
                                 ('No3', '<f8'), ('No4', '<f8')])
    sarray = np.zeros(len(ran_int), dtype=dty)
    sarray['Date'] = dt.datetime.now()
    sarray['No1'] = ran_int[:, 0]
    sarray['No2'] = ran_int[:, 1]
    sarray['No3'] = ran_flo[:, 0]
    sarray['No4'] = ran_flo[:, 1]
    h5.create_table('/', 'ints_floats_from_array', sarray,
                      title='Integers and Floats',
                      expectedrows=rows, filters=filters)
    print(h5)
    h5.remove_node('/', 'ints_floats_from_array')
    print(tab[:3])
    print(tab[:4]['No4'])
    print(np.sum(tab[:]['No3']))
    print(np.sum(np.sqrt(tab[:]['No1'])))
    plt.hist(tab[:]['No3'], bins=30)
    plt.grid(True)
    print(len(tab[:]['No3']))
    plt.plot()
    plt.savefig(PNG_PATH + 'h5.png', dpi=300)
    plt.close()
    
    res = np.array([(row['No3'], row['No4']) for row in
        tab.where('((No3 < -0.05) | (No3 > 0.05)) \
                 & ((No4 < -0.1) | (No4 > 0.1))')])[::100]
    plt.plot(res.T[0], res.T[1], 'ro')
    plt.grid(True)
    plt.savefig(PNG_PATH + 'h5_x.png', dpi=300)
    plt.close()
    
    values = tab.cols.No3[:]
    print("Max %18.3f" % values.max())
    print("Ave %18.3f" % values.mean())
    print("Min %18.3f" % values.min())
    print("Std %18.3f" % values.std())
    results = [(row['No1'], row['No2']) for row in
               tab.where('((No1 > 9800) | (No1 < 200)) \
                        & ((No2 > 4500) & (No2 < 5500))')]
    for res in results[:4]:
        print(res)
        
    results = [(row['No1'], row['No2']) for row in
           tab.where('(No1 == 1234) & (No2 > 9776)')]
    for res in results:
        print(res)
    
    filename = path + 'tab.h5c'
    h5c = tb.open_file(filename, 'w')
    filters = tb.Filters(complevel=4, complib='blosc')
    tabc = h5c.create_table('/', 'ints_floats', sarray,
                            title='Integers and Floats',
                          expectedrows=rows, filters=filters)
    res = np.array([(row['No3'], row['No4']) for row in
                 tabc.where('((No3 < -0.5) | (No3 > 0.5)) \
                           & ((No4 < -1) | (No4 > 1))')])[::100]
    arr_non = tab.read()
    arr_com = tabc.read()
    h5c.close()
    
    arr_int = h5.create_array('/', 'integers', ran_int)
    arr_flo = h5.create_array('/', 'floats', ran_flo)
    print(h5)
    h5.close()
    
    filename = path + 'array.h5'
    h5 = tb.open_file(filename, 'w')
    
    n = 100
    ear = h5.create_earray(h5.root, 'ear',
                      atom=tb.Float64Atom(),
                      shape=(0, n))
    rand = np.random.standard_normal((n, n))
    for i in range(750):
        ear.append(rand)
    ear.flush()
    print(ear)
    print(ear.size_on_disk)
    out = h5.create_earray(h5.root, 'out',
                      atom=tb.Float64Atom(),
                      shape=(0, n))
    expr = tb.Expr('3 * sin(ear) + sqrt(abs(ear))')
    expr.set_output(out, append_mode=True)
    print(expr.eval())
    imarray = ear.read()
    
    import numexpr as ne
    expr = '3 * sin(imarray) + sqrt(abs(imarray))'
    ne.set_num_threads(16)
    print(ne.evaluate(expr)[0, :10])
    h5.close()
    

if __name__ == "__main__":
    # pkl()
    # read_write()
    # sql_db()
    # write_read_np_arrs()
    pandas_io()