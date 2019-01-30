import oandapy
import pdb
from datetime import datetime, timedelta

# Enter your account ID and API key here.
account_id = 10401939
key = '2705786cbcd782c9048f3ad145d95501-19cc3c9def75c4b35fa7f00a5f4478cb'


""" Fetch rates """
# oanda = oandapy.API(environment="sandbox", access_token=key)
oanda = oandapy.API(environment="practice", access_token=key)
response = oanda.get_prices(instruments="EUR_USD")
print(response)

prices = response["prices"]
bidding_price = float(prices[0]["bid"])
asking_price = float(prices[0]["ask"])
instrument = prices[0]["instrument"]
time = prices[0]["time"]
print("[%s] %s bid=%s ask=%s" % (time, instrument, bidding_price, asking_price))

""" Send an order """
# set the trade to expire after one day
trade_expire = datetime.now() + timedelta(days=1)
trade_expire = trade_expire.isoformat("T") + "Z"

response = oanda.create_order(
    account_id,
    instrument="EUR_USD",
    units=1000,
    side="sell",  # "buy" or "sell"
    type="limit",
    price=1.105,
    expiry=trade_expire)
print(response)