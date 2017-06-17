import sys, pdb
sys.path.append('/usr/share/doc')
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
# import statsmodels.api as sm
import scipy.interpolate as spi
import scipy.optimize as spo
import scipy.integrate as sci
from math import sqrt
from matplotlib.patches import Polygon
import sympy as sy

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

def noisy():
    xn = np.linspace(-2 * np.pi, 2 * np.pi, 50)
    xn = xn + 0.15 * np.random.standard_normal(len(xn))
    yn = f(xn) + 0.25 * np.random.standard_normal(len(xn))
    reg = np.polyfit(xn, yn, 7)
    ry = np.polyval(reg, xn)
    plt.plot(xn, yn, 'b^', label='f(x)')
    plt.plot(xn, ry, 'ro', label='regression')
    plt.legend(loc=0)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig(PATH + 'noisy.png', dpi=300)
    plt.close()

def unsorted():
    xu = np.random.rand(50) * 4 * np.pi - 2 * np.pi
    yu = f(xu)
    print(xu[:10].round(2))
    print(yu[:10].round(2))
    reg = np.polyfit(xu, yu, 5)
    ry = np.polyval(reg, xu)
    plt.plot(xu, yu, 'b^', label='f(x)')
    plt.plot(xu, ry, 'ro', label='regression')
    plt.legend(loc=0)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig(PATH + 'unsorted.png', dpi=300)
    plt.close()
    
def fm(p):
    x, y = p
    return np.sin(x) + 0.25 * x + np.sqrt(y) + 0.05 * y ** 2

def multi_d():
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    X, Y = np.meshgrid(x, y)
    # generates 2-d grids out of the 1-d arrays
    Z = fm((X, Y))
    x = X.flatten()
    y = Y.flatten()
    # yields 1-d arrays from the 2-d grids

    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=mpl.cm.coolwarm,
        linewidth=0.5, antialiased=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(PATH + 'multi_d.png', dpi=300)
    plt.close()
    
    matrix = np.zeros((len(x), 6 + 1))
    matrix[:, 6] = np.sqrt(y)
    matrix[:, 5] = np.sin(x)
    matrix[:, 4] = y ** 2
    matrix[:, 3] = x ** 2
    matrix[:, 2] = y
    matrix[:, 1] = x
    matrix[:, 0] = 1
    
    pdb.set_trace()
    regr = linear_model.LinearRegression()
    regr.fit(fm((x, y)), (np.transpose(matrix)))
    print(model.rsquared)
    a = model.params
    print(a)
    
    RZ = reg_func(a, (X, Y))
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca(projection='3d')
    surf1 = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
                cmap=mpl.cm.coolwarm, linewidth=0.5,
                antialiased=True)
    surf2 = ax.plot_wireframe(X, Y, RZ, rstride=2, cstride=2,
                              label='regression')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.legend()
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(PATH + 'multi_d2.png', dpi=300)
    plt.close()
    
def reg_func(a, p):
    x, y = p
    f6 = a[6] * np.sqrt(y)
    f5 = a[5] * np.sin(x)
    f4 = a[4] * y ** 2
    f3 = a[3] * x ** 2
    f2 = a[2] * y
    f1 = a[1] * x
    f0 = a[0] * 1
    return (f6 + f5 + f4 + f3 +
            f2 + f1 + f0)

def interpolation():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 25)
    ipo = spi.splrep(x, ff(x), k=1)
    iy = spi.splev(x, ipo)
    plt.plot(x, ff(x), 'b', label='f(x)')
    plt.plot(x, iy, 'r.', label='interpolation')
    plt.legend(loc=0)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig(PATH + 'interp.png', dpi=300)
    plt.close()
    print(np.allclose(ff(x), iy))
    
    xd = np.linspace(1.0, 3.0, 50)
    iyd = spi.splev(xd, ipo)
    plt.plot(xd, ff(xd), 'b', label='f(x)')
    plt.plot(xd, iyd, 'r.', label='interpolation')
    plt.legend(loc=0)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig(PATH + 'interp2.png', dpi=300)
    plt.close()
    
    ipo = spi.splrep(x, ff(x), k=3)
    iyd = spi.splev(xd, ipo)
    plt.plot(xd, ff(xd), 'b', label='f(x)')
    plt.plot(xd, iyd, 'r.', label='interpolation')
    plt.legend(loc=0)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig(PATH + 'interp3.png', dpi=300)
    plt.close()
    print(np.allclose(ff(xd), iyd))
    print(np.sum((ff(xd) - iyd) ** 2) / len(xd))
    
def convex_optimization():
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    Z = ffm(X, Y)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=mpl.cm.coolwarm,
            linewidth=0.5, antialiased=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(PATH + 'convex.png', dpi=300)
    plt.close()
    
def ffm(x, y):
    return (np.sin(x) + 0.05 * x ** 2
          + np.sin(y) + 0.05 * y ** 2)    

def ff(x):
    return np.sin(x) + 0.5 * x

def fo(x):
    x, y = x
    output = False
    z = np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2
    if output == True:
        print('%8.4f %8.4f %8.4f' % (x, y, z))
    return z

def global_opt():
    spo.brute(fo, ((-10, 10.1, 5), (-10, 10.1, 5)), finish=None)
    opt1 = spo.brute(fo, ((-10, 10.1, 0.1), (-10, 10.1, 0.1)), finish=None)
    print(opt1)
    print(ffm(opt1[0], opt1[1]))
    
def local_opt():
    opt1 = spo.brute(fo, ((-10, 10.1, 0.1), (-10, 10.1, 0.1)), finish=None)
    opt2 = spo.fmin(fo, opt1, xtol=0.001, ftol=0.001, maxiter=15, maxfun=20)
    print(ffm(opt2[0], opt2[1]))
    print(spo.fmin(fo, (2.0, 2.0), maxiter=250))

def Eu(s_b):
    return -(0.5 * sqrt(s_b[0] * 15 + s_b[1] * 5) + 0.5 * sqrt(s_b[0] * 5 + s_b[1] * 12))

def constrained_opt():
    # constraints
    cons = ({'type': 'ineq', 'fun': lambda s_b:  100 - s_b[0] * 10 - s_b[1] * 10})
    # budget constraint
    bnds = [[0, 1000], [0, 1000]]  # uppper bounds large enough
    result = spo.minimize(Eu, [5, 5], method='SLSQP',
                       bounds=bnds, constraints=cons)
    print(result)
    print(result['x'])
    print(-result['fun'])
    print(np.dot(result['x'], [10, 10]))

def fff(x):
    return np.sin(x) + 0.5 * x

def integration():
    a = 0.5  # left integral limit
    b = 9.5  # right integral limit
    x = np.linspace(0, 10)
    y = fff(x)

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.plot(x, y, 'b', linewidth=2)
    plt.ylim(ymin=0)

    # area under the function
    # between lower and upper limit
    Ix = np.linspace(a, b)
    Iy = fff(Ix)
    verts = [(a, 0)] + list(zip(Ix, Iy)) + [(b, 0)]
    poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
    ax.add_patch(poly)

    # labels
    plt.text(0.75 * (a + b), 1.5, r"$\int_a^b f(x)dx$",
             horizontalalignment='center', fontsize=20)
    plt.figtext(0.9, 0.075, '$x$')
    plt.figtext(0.075, 0.9, '$f(x)$')
    ax.set_xticks((a, b))
    ax.set_xticklabels(('$a$', '$b$'))
    ax.set_yticks([fff(a), fff(b)])
    plt.savefig(PATH + 'integration.png', dpi=300)
    plt.close()

def numerical_int():
    a = 0.5  # left integral limit
    b = 9.5  # right integral limit
    x = np.linspace(0, 10)
    y = fff(x)
    print(sci.fixed_quad(fff, a, b)[0])
    print(sci.quad(fff, a, b)[0])
    print(sci.romberg(fff, a, b))
    xi = np.linspace(0.5, 9.5, 25)
    print(sci.trapz(fff(xi), xi))
    print(sci.simps(fff(xi), xi))

def int_simulation():
    a = 0.5  # left integral limit
    b = 9.5  # right integral limit
    x = np.linspace(0, 10)
    y = fff(x)
    
    for i in range(1, 20):
        np.random.seed(1000)
        x = np.random.random(i * 10) * (b - a) + a
        print(np.sum(fff(x)) / len(x) * (b - a))

def sym_comp_basics():
    x = sy.Symbol('x')
    y = sy.Symbol('y')
    print(type(x))
    print(sy.sqrt(x))
    print(3 + sy.sqrt(x) - 4 ** 2)
    f = x ** 2 + 3 + 0.5 * x ** 2 + 3 / 2
    print(sy.simplify(f))
    sy.init_printing(pretty_print=False, use_unicode=False)
    print(sy.pretty(f))
    print(sy.pretty(sy.sqrt(x) + 0.5))
    pi_str = str(sy.N(sy.pi, 400000))
    print(pi_str[:40])
    print(pi_str[-40:])
    print(pi_str.find('111272'))

def sym_comp_eqs():
    x = sy.Symbol('x')
    y = sy.Symbol('y')
    print(sy.solve(x ** 2 - 1))
    print(sy.solve(x ** 2 - 1 - 3))
    print(sy.solve(x ** 3 + 0.5 * x ** 2 - 1))
    print(sy.solve(x ** 2 + y ** 2))

def sym_comp_int():
    x = sy.Symbol('x')
    a, b = sy.symbols('a b')
    print(sy.pretty(sy.Integral(sy.sin(x) + 0.5 * x, (x, a, b))))
    int_func = sy.integrate(sy.sin(x) + 0.5 * x, x)
    print(sy.pretty(int_func))
    Fb = int_func.subs(x, 9.5).evalf()
    Fa = int_func.subs(x, 0.5).evalf()
    print(Fb - Fa)  # exact value of integral
    int_func_limits = sy.integrate(sy.sin(x) + 0.5 * x, (x, a, b))
    print(sy.pretty(int_func_limits))
    print(int_func_limits.subs({a : 0.5, b : 9.5}).evalf())
    print(sy.integrate(sy.sin(x) + 0.5 * x, (x, 0.5, 9.5)))

def sym_comp_diff():
    x = sy.Symbol('x')
    y = sy.Symbol('y')
    int_func = sy.integrate(sy.sin(x) + 0.5 * x, x)
    print(int_func.diff())
    f = (sy.sin(x) + 0.05 * x ** 2
        + sy.sin(y) + 0.05 * y ** 2)
    del_x = sy.diff(f, x)
    print(del_x)
    del_y = sy.diff(f, y)
    print(del_y)
    xo = sy.nsolve(del_x, -1.5)
    print(xo)
    yo = sy.nsolve(del_y, -1.5)
    print(yo)
    print(f.subs({x : xo, y : yo}).evalf())
    xo = sy.nsolve(del_x, 1.5)
    print(xo)
    yo = sy.nsolve(del_y, 1.5)
    print(yo)
    print(f.subs({x : xo, y : yo}).evalf())


if __name__ == '__main__':
    # approx()
    # regress()
    # noisy()
    # unsorted()
    # multi_d()
    # interpolation()
    # convex_optimization()
    # global_opt()
    # local_opt()
    # constrained_opt()
    # integration()
    # numerical_int()
    # int_simulation()
    # sym_comp_basics()
    sym_comp_eqs()
    sym_comp_int()
    sym_comp_diff()