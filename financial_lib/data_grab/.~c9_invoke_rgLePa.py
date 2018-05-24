import sys
sys.path.append("/home/ubuntu/workspace/finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import pdb, requests, datetime, csv
import pandas as pd
from financial_lib.futures_forwards.data_grab.quandl_api_helper import quandl_api_dict, URL, MONTH_MAP

def callQuandlAPI(call_url):
    urlData = requests.get(call_url).content.decode('utf-8')
    cr = csv.reader(urlData.splitlines(), delimiter=',')
    data = []
    for row in list(cr):
        data.append(row)
    df = pd.DataFrame(data)
    if df[0][0] == 'code':
        raise Exception('No contract for this month')
    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop(0))
    return df
    
if __name__ == '__main__':
    # callQuandlAPI()
    pass