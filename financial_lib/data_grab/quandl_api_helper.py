# FORMAT: name : [database_code, dataseries_code]
from utils.utils import QUANDL_API_KEY

# URL = 'https://www.quandl.com/api/v3/datasets/{0}/{1}/data.csv?start_date={2}&end_date={3}&api_key=' + QUANDL_API_KEY
# URL = 'https://www.quandl.com/api/v3/datasets/{0}/{1}/data.csv?api_key=' + QUANDL_API_KEY
# CHRIS_URL = 'https://www.quandl.com/api/v3/datasets/CHRIS/{0}.csv?start_date={1}&end_date={2}&api_key=J4d6zKiPjebay-zW7T8X'

URLs = {
    'CME' : 'https://www.quandl.com/api/v3/datasets/CME/{0}/data.csv?api_key=' + QUANDL_API_KEY,
    'ICE' : 'https://www.quandl.com/api/v3/datasets/ICE/{0}/data.csv?api_key=' + QUANDL_API_KEY,
    'CHRIS' : 'https://www.quandl.com/api/v3/datasets/CHRIS/{0}.csv?start_date={1}&end_date={2}&api_key=' + QUANDL_API_KEY
}

# quandl_api_dict = {
#     'corn' : ['CME', 'C'],
#     'gold' : ['CME', 'GC'],
#     'brent_oil' : ['CME', 'BB'],
#     'wti_oil' : ['CME', 'T'],
#     'sugar' : ['CME', 'YO'],
# }

quandl_api_dict = {
    'corn' : ['ICE', 'C'],
    'gold' : ['ICE', 'GC'],
    'brent_oil' : ['ICE', 'B'],
    'wti_oil' : ['ICE', 'T'],
    'sugar' : ['ICE', 'YO'],
    'EUR' : ['CME', 'EC'],
    'JPY' : ['CME', 'JY'],
    'GBP' : ['CME', 'BP'],
}

quandl_api_hist_dict = {
    'gold' : ['CHRIS', 'CME_GC1']
}


MONTH_MAP = {
    1 : 'F',
    2 : 'G',
    3 : 'H',
    4 : 'J',
    5 : 'K',
    6 : 'M',
    7 : 'N',
    8 : 'Q',
    9 : 'U',
    10 : 'V',
    11 : 'X',
    12 : 'Z'
}