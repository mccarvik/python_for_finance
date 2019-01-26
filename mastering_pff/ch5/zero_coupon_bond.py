def zero_coupon_bond(par, y, t):
    """
    Price a zero coupon bond.
    
    Spot rates directly observable in the market
    Zero rates are extracted from some asset to serve as a proxy for the spot rate
    same for pretty much all purposes, but there is that distinction
    
    Par - face value of the bond.
    y - annual yield or rate of the bond.
    t - time to maturity in years.
    
    To calculate spot rate:
    px = (par (usuallu 100)) / (e^(t * y))
    usually have px, know time to maturity t, and solve for y
    """
    return par/(1+y)**t

print(zero_coupon_bond(100, 0.05, 5))