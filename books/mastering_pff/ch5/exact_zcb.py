import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math

IMG_PATH = "/home/ubuntu/workspace/python_for_finance/mastering_pff/png/"

""" Get zero coupon bond price by Vasicek model """
def exact_zcb(theta, kappa, sigma, tau, r0=0.):
    # rewries the zero coupon bond price based on rate expectations according to the vasicek model
    B = (1 - np.exp(-kappa*tau)) / kappa
    A = np.exp((theta-(sigma**2)/(2*(kappa**2))) * (B-tau) - (sigma**2)/(4*kappa)*(B**2))
    return A * np.exp(-r0*B)


def exercise_value(K, R, t):
    # K is ratio of the strike price to the par value (static over the life of the bond)
    # discount that value to today
    return K*math.exp(-R*t)


if __name__ == "__main__":
    Ts = np.r_[0.0:25.5:0.5]
    zcbs = [exact_zcb(0.5, 0.02, 0.03, t, 0.015) for t in Ts]

    plt.title("Zero Coupon Bond (ZCB) Values by Time")
    plt.plot(Ts, zcbs, label='ZCB')
    plt.ylabel("Value ($)")
    plt.xlabel("Time in years")
    plt.legend()
    plt.grid(True)
    plt.savefig(IMG_PATH + 'zcb.png', dpi=300)
    plt.close()

    # callable bond will be the min of the strike and th ZCB at each point in time
    # since issuer owns the calls
    Ks = [exercise_value(0.95, 0.015, t) for t in Ts]
    plt.title("Zero Coupon Bond (ZCB) "
              "and Strike (K) Values by Time")
    plt.plot(Ts, zcbs, label='ZCB')
    plt.plot(Ts, Ks, label='K', linestyle="--", marker=".")
    plt.ylabel("Value ($)")
    plt.xlabel("Time in years")
    plt.legend()
    plt.grid(True)
    plt.savefig(IMG_PATH + 'zcb_and_strike.png', dpi=300)
    plt.close()