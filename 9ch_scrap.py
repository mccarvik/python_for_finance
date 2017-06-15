import sys
sys.path.append("/usr/local/lib/python3.4/dist-packages")
sys.path.append("/usr/local/lib/python3.5/dist_packages")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns; sns.set()
mpl.rcParams['font.family'] = 'serif'

PATH = '/home/ubuntu/workspace/python_for_finance/png/ch9/'

def approx():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 50)
    plt.plot(x, f(x), 'b')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig(PATH + 'approx.png', dpi=300)
    plt.close()

def f(x):
    return np.sin(x) + 0.5 * x
    
def regress():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 50)
    reg = np.polyfit(x, f(x), deg=1)
    ry = np.polyval(reg, x)
    plt.plot(x, f(x), 'b', label='f(x)')
    plt.plot(x, ry, 'r.', label='regression')
    plt.legend(loc=0)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig(PATH + 'reg1.png', dpi=300)
    plt.close()
    
    reg = np.polyfit(x, f(x), deg=5)
    ry = np.polyval(reg, x)
    plt.plot(x, f(x), 'b', label='f(x)')
    plt.plot(x, ry, 'r.', label='regression')
    plt.legend(loc=0)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig(PATH + 'reg2.png', dpi=300)
    plt.close()
    
    reg = np.polyfit(x, f(x), deg=7)
    ry = np.polyval(reg, x)
    plt.plot(x, f(x), 'b', label='f(x)')
    plt.plot(x, ry, 'r.', label='regression')
    plt.legend(loc=0)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig(PATH + 'reg3.png', dpi=300)
    plt.close()
    
    print(np.allclose(f(x), ry))
    print(np.sum((f(x) - ry) ** 2) / len(x))

if __name__ == '__main__':
    # approx()
    regress()