""" Calculate modified duration of a bond """
from bond_ytm import bond_ytm
from bond_price import bond_price


def bond_mod_duration(price, par, T, coup, freq, dy=0.01):
    ytm = bond_ytm(price, par, T, coup, freq)

    ytm_minus = ytm - dy    
    price_minus = bond_price(par, T, ytm_minus, coup, freq)
    
    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coup, freq)
    
    # calculate the difference in price when bond moves up and down
    # for a given change in yield
    mduration = (price_minus-price_plus)/(2.*price*dy)
    return mduration

if __name__ == "__main__":
    print(bond_mod_duration(95.04, 100, 1.5, 5.75, 2, 0.01))