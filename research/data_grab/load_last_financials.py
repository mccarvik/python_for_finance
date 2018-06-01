import sys
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import pdb, time, requests, csv
import numpy as np
import pandas as pd
import datetime as dt

success = []
failure = []


def getData(tickers=None):
    # Getting all the tickers from text file
    tasks = []
    if not tickers:
        with open("/home/ubuntu/workspace//ml_dev_work/utils/snp_ticks_2018_02_13.txt", "r") as f:
            for line in f:
                tasks.append(line.strip())
    else:
        tasks = tickers
    t0 = time.time()
    threads = []
    # tasks = [t for t in tasks if t not in ms_dg_helper.remove_ticks_ms]
    # tasks = [t for t in tasks if t not in ms_dg_helper.remove_ticks_dr]
    
    try:
        ct = 0
        for t in tasks:
            if ct % 25 == 0:
                print(str(ct) + " stocks completed so far")
            try:
                makeAPICall(t)
                success.append(t)
            except:
                failure.append(t)
                print("Failed " + t + "\t")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print("Error in task loop: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
            ct+=1
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Error in getData: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
    
    t1 = time.time()
    text_file = open("Failures.txt", "w")
    text_file.write(("\t").join(failure))
    print("\t".join(success))
    print("Done Retrieving data, took {0} seconds".format(t1-t0))