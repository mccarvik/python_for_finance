# FORMAT: name : [database_code, dataseries_code]
from utils.utils import QUANDL_API_KEY

# URL = 'https://www.quandl.com/api/v3/datasets/{0}/{1}/data.csv?start_date={2}&end_date={3}&api_key=' + QUANDL_API_KEY
URL = 'https://www.quandl.com/api/v3/datasets/{0}/{1}/data.csv?api_key=' + QUANDL_API_KEY
# CHRIS DB if needed:  https://www.quandl.com/api/v3/datasets/CHRIS/CME_C2.csv?api_key=J4d6zKiPjebay-zW7T8X

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
    'brent_oil' : ['ICE', 'BB'],
    'wti_oil' : ['ICE', 'T'],
    'sugar' : ['ICE', 'YO'],
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