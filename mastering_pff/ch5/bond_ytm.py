""" Get yield-to-maturity of a bond """
import scipy.optimize as optimize


def bond_ytm(price, par, T, coup, freq=2, guess=0.05):
    freq = float(freq)
    periods = T*freq
    coupon = coup/100.*par/freq
    dt = [(i+1)/freq for i in range(int(periods))]
    ytm_func = lambda y: sum([coupon/(1+y/freq)**(freq*t) for t in dt]) + par/(1+y/freq)**(freq*T) - price
    # Uses a newton-raphson approximation to find the ytm root
    return optimize.newton(ytm_func, guess)

if __name__ == "__main__":
    ytm = bond_ytm(95.0428, 100, 1.5, 5.75, 2)
    print(ytm)
    