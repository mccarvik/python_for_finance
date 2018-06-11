import sys
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import pdb, time, requests, csv
import numpy as np
import pandas as pd
import datetime as dt
from research.res_utils import *
from utils.db_utils import DBHelper, restart

success = []
failure = []


def getData(tickers=None):
    # Getting all the tickers from text file
    tasks = []
    if not tickers:
        # with open("/home/ubuntu/workspace//ml_dev_work/utils/snp_ticks_2018_02_13.txt", "r") as f:
        with open("/home/ubuntu/workspace//ml_dev_work/utils/memb_list.txt", "r") as f:
            for line in f:
                tasks.append(line.strip())
    else:
        tasks = tickers
    t0 = time.time()
    threads = []
    # tasks = [t for t in tasks if t not in ms_dg_helper.remove_ticks_ms]
    # tasks = [t for t in tasks if t not in ms_dg_helper.remove_ticks_dr]
    
    with DBHelper() as db:
        db.connect()
        already = db.select('morningstar_monthly_cf')['ticker'].values
    
    try:
        ct = 0
        for t in tasks:
            if t in already:
                continue
            if ct % 25 == 0:
                print(str(ct) + " stocks completed so far")
            try:
                succ = True
                # data = makeAPICall(t, 'is')
                # if len(data) != 0:
                #     sendToDB(data, 'morningstar_monthly_is')
                # else:
                #     succ = False
                #     failure.append(t)
                
                # data = makeAPICall(t, 'bs')
                # if len(data) != 0:
                #     sendToDB(data, 'morningstar_monthly_bs')
                # else:
                #     succ = False
                #     failure.append(t)
                
                data = makeAPICall(t, 'cf')
                if len(data) != 0:
                    sendToDB(data, 'morningstar_monthly_cf')
                else:
                    succ = False
                    failure.append(t)
                
                if succ:
                    success.append(t)
                
                # data = makeAPICall(t, 'cf')
                # sendToDB(data, 'morningstar_monthly_cf')
                
            except:
                # pdb.set_trace()
                failure.append(t)
                print("Failed " + t + "\t")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                # print("Error in task loop: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
            ct+=1
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Error in getData: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
    
    t1 = time.time()
    text_file = open("Failures.txt", "w")
    text_file.write(("\t").join(failure))
    print("\t".join(success))
    print("Done Retrieving data, took {0} seconds".format(t1-t0))
    

def sendToDB(df, table):
    df = df.reset_index()[df.reset_index()['date'] != 'TTM'].set_index(idx)
    with DBHelper() as db:
        db.connect()
        prim_keys = ['date', 'ticker', 'month']
        for ind, vals in df.reset_index().iterrows():
            val_dict = vals.to_dict()
            db.upsert(table, val_dict, prim_keys)


if __name__ == "__main__":
    # getData(['MCD'])
    getData()