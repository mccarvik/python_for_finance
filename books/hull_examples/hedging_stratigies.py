import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

# import numpy as np
# import pandas as pd
import datetime as dt

# from dx import *

# Value that minimizes the variance between two products when cross hedging
def minimum_variance_hedge_ratio(corr, stdev_spot, stdev_fut):
    return corr * (stdev_spot / stdev_fut)
    

# Calculating the optimal numebr of contracts to buy for hedging
def optimal_number_contracts(hedge_ratio, size_pos_hedged, size_fut_cont):
    return (hedge_ratio * size_pos_hedged) / size_fut_cont


# How many contracts when hedging an equity portfolio
def equity_portfolio_hedge(val_of_port, val_of_fut, port_beta):
    return port_beta * (val_of_port / val_of_fut)
    
# Calculates the expected return on an asset
def capital_asset_pricing_model(rf, beta, ret_market):
    return rf + beta * (ret_market - rf)


if __name__ == '__main__':
    jet_fuel_spot_stdev = 0.0263
    heating_oil_fut_stdev = 0.0313
    corr = 0.928
    h = minimum_variance_hedge_ratio(corr, jet_fuel_spot_stdev, heating_oil_fut_stdev)
    print("min hedge ratio: " + str(h))
    jet_fuel_cont_size = 2000000
    heating_oil_cont_size = 42000
    print("Number of Contracts: " + str(optimal_number_contracts(h, jet_fuel_cont_size, heating_oil_cont_size)))
    
    beta = 1.5
    port_val = 5050000
    fut_val = 252500
    print("Number of Contracts: " + str(equity_portfolio_hedge(port_val, fut_val, beta)))
    
    print("Expected Return: " + str(capital_asset_pricing_model(0.05, 0.75, 0.13)))