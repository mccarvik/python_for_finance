""" Calculate convexity of a bond """
from bond_ytm import bond_ytm
from bond_price import bond_price


def bond_convexity(price, par, T, coup, freq, dy=0.01):
    """
    Measures the change in duration due to a change in yield
    a convexity of 300 means a 1% change in yield will lead to a 3% (0.03) change in price resulting from convexity
    """
    ytm = bond_ytm(price, par, T, coup, freq)

    ytm_minus = ytm - dy    
    price_minus = bond_price(par, T, ytm_minus, coup, freq)
    
    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coup, freq)
    
    convexity = (price_minus+price_plus-2*price)/(price*dy**2)
    return convexity

if __name__ == "__main__":
    print(bond_convexity(95.0428, 100, 1.5, 5.75, 2))