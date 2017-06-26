import pdb
import seaborn as sns; sns.set()
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
import http.client as clt
import urllib.request


def web_basics_http():
    http = clt.HTTPConnection('hilpisch.com')
    http.request('GET', '/index.htm')
    resp = http.getresponse()
    print(resp.status, resp.reason)
    content = resp.read()
    print(content[:100])
    index = content.find(b' E ')
    print(index)
    print(content[index:index + 29])
    http.close()

def web_basics_urllib():
    pdb.set_trace()
    url = 'http://ichart.finance.yahoo.com/table.csv?g=d&ignore=.csv'
    url += '&s=YHOO&a=01&b=1&c=2014&d=02&e=6&f=2014'
    connect = urllib.request.urlopen(url)
    data = connect.read()
    print(data)
    url = 'http://ichart.finance.yahoo.com/table.csv?g=d&ignore=.csv'
    url += '&%s'  # for replacement with parameters
    url += '&d=06&e=30&f=2014'
    params = urllib.parse.urlencode({'s': 'MSFT', 'a': '05', 'b': 1, 'c': 2014})
    print(params)
    print(url % params)
    connect = urllib.request.urlopen(url % params)
    data = connect.read()
    print(data)
    print(urllib.request.urlretrieve(url % params, './data/msft.csv'))
    csv = open('./data/msft.csv', 'r')
    csv.readlines()[:5]

if __name__ == '__main__':
    # web_basics_http()
    web_basics_urllib()