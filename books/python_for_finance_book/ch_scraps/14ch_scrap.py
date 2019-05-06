import sys, pdb
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb, ftplib, http, http.client
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import urllib, warnings, time, requests
import plotly.plotly as py
import cufflinks as cf
from pandas_datareader.data import DataReader
import datetime as dt
from jinja2 import Template

PATH = '/home/ubuntu/workspace/python_for_finance/png/book_examples/ch14/'

def web_basics_ftplib():
    ftp = ftplib.FTP('quant-platform.com')
    print(ftp.login(user='python', passwd='python'))
    np.save('./data/array', np.random.standard_normal((100, 100)))
    f = open('./data/array.npy', 'r')
    print(ftp.storbinary('STOR array.npy', f))
    print(ftp.retrlines('LIST'))
    f = open('./data/array_ftp.npy', 'wb').write
    print(ftp.retrbinary('RETR array.npy', f))
    print(ftp.delete('array.npy'))
    print(ftp.retrlines('LIST'))
    ftp.close()
    ftps = ftplib.FTP_TLS('quant-platform.com')
    print(ftps.login(user='python', passwd='python'))
    print(ftps.prot_p())
    print(ftps.retrlines('LIST'))
    ftps.close()

def web_basics_http():
    h = http.client.HTTPConnection('hilpisch.com')
    h.request('GET', '/index.htm')
    resp = h.getresponse()
    print(resp.status, resp.reason)
    content = resp.read()
    print(content[:100])
    # first 100 characters of the file
    index = content.find(b' E ')
    print(index)
    print(content[index:index + 29])
    h.close()

# Doesnt work any more as yahoo no longer provides this service but urllib use case still valid
def web_basics_urllib():
    url = 'http://ichart.finance.yahoo.com/table.csv?g=d&ignore=.csv'
    url += '&s=YHOO&a=01&b=1&c=2014&d=02&e=6&f=2014'
    connect = urllib.urlopen(url)
    data = connect.read()
    print(data)
    url = 'http://ichart.finance.yahoo.com/table.csv?g=d&ignore=.csv'
    url += '&%s'  # for replacement with parameters
    url += '&d=06&e=30&f=2014'
    params = urllib.urlencode({'s': 'MSFT', 'a': '05', 'b': 1, 'c': 2014})
    print(params)
    print(url % params)
    connect = urllib.urlopen(url % params)
    data = connect.read()
    print(data)
    print(urllib.urlretrieve(url % params, './data/msft.csv'))
    csv = open('./data/msft.csv', 'r')
    csv.readlines()[:5]

def web_plotting_plots():
    # static
    start = dt.datetime(2010, 1, 1)
    end = dt.datetime(2013, 1, 27)
    data=DataReader("MSFT", 'google', start, end)
    data = data.reset_index()
    fig, ax = plt.subplots()
    data.plot(x='Date', y='Close', grid=True, ax=ax)
    plt.savefig(PATH + 'MSFT.png', dpi=300)
    
    # interactive - may not work in cloud 9
    warnings.simplefilter('ignore')
    py.sign_in('Python-Demo-Account', 'gwt101uhh0')
    # to interactive D3.js plot
    py.iplot_mpl(fig)
    # direct approach with Cufflinks
    data.set_index('Date')['Close'].iplot(world_readable=True)

def web_plotting_realtime():
    pdb.set_trace()
    # Site now needs log in
    url = 'http://analytics.quant-platform.com:12500/prices'
    # real-time FX (dummy!) data from JSON API
    api = requests.get(url)
    print(api)
    data = api.json()
    print(data)
    print(data['tick']['ask'])
    ticks = pd.DataFrame({'bid': data['tick']['bid'],
                      'ask': data['tick']['ask'],
                      'instrument': data['tick']['instrument'],
                      'time': pd.Timestamp(data['tick']['time'])},
                      index=[pd.Timestamp(data['tick']['time']),])
    # initialization of ticks DataFrame
    print(ticks[['ask', 'bid', 'instrument']])
    
    # URL also no longer valid
    url1 = 'http://www.netfonds.no/quotes/posdump.php?'
    url2 = 'date=%s%s%s&paper=%s.N&csv_format=csv'
    url = url1 + url2
    # must be a business day
    today = dt.datetime.now()
    y = '%d' % today.year
    # current year
    m = '%02d' % today.month
    # current month, add leading zero if needed
    d = '%02d' % today.day
    # current day, add leading zero if needed
    sym = 'NKE'
    print(y, m, d, sym)
    urlreq = url % (y, m, d, sym)
    print(urlreq)
    data = pd.read_csv(urlreq, parse_dates=['time'])
    # initialize DataFrame object
    print(data.info())

def templating():
    print('%d, %d, %d' % (1, 2, 3))
    print('{}, {}, {}'.format(1, 2, 3))
    print('{}, {}, {}'.format(*'123'))
    templ = '''<!doctype html>
        Just print out <b>numbers</b> provided to the template.
        <br><br>
        {% for number in numbers %}
            {{ number }}
        {% endfor %}
        '''
    t = Template(templ)
    html = t.render(numbers=range(5))
    print(html)


if __name__ == '__main__':
    # web_basics_ftplib()
    # web_basics_http()
    # web_basics_urllib()
    # web_plotting_plots()
    # web_plotting_realtime()
    templating()