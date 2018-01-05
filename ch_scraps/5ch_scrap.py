import matplotlib as mpl
mpl.use('Agg')
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
# from matplotlib.finance import quotes_historical_yahoo_ohlc
import matplotlib.finance as mpf
from mpl_toolkits.mplot3d import Axes3D

PATH = '/home/ubuntu/workspace/python_for_finance/png/book_examples/ch5/'

def one_dims():
    np.random.seed(1000)
    y = np.random.standard_normal(20)
    x = range(len(y))
    plt.plot(x, y)
    plt.savefig(PATH + 'rando_plot.png', dpi=300)
    plt.close()
    
    x = range(len(y))
    plt.plot(y.cumsum())
    plt.savefig(PATH + 'cumsum.png', dpi=300)
    plt.close()
    
    x = range(len(y))
    plt.plot(y.cumsum())
    plt.grid(True)
    plt.axis('tight')
    plt.xlim(-1, 20)
    plt.ylim(np.min(y.cumsum()) - 1,
            np.max(y.cumsum()) + 1)
    plt.savefig(PATH + 'cumsum_grid.png', dpi=300)
    plt.close()
    
def two_dims():
    np.random.seed(2000)
    y = np.random.standard_normal((20,2)).cumsum(axis=0)
    y[:, 0] = y[:, 0] * 100
    # key line, gives us axis to the figure and axises separately
    fig, ax1 = plt.subplots()
    # plt.figure(figsize=(7,4))
    plt.plot(y[:, 0], 'b', lw=1.5, label='1st')
    # ro = red dots
    plt.plot(y[:, 0], 'ro')
    plt.legend(loc=8)
    plt.axis('tight')
    plt.xlabel('index')
    plt.ylabel('value 1st')
    plt.title('A Simple Plot')
    
    # Key line below - get a second plot that shares the x axis
    ax2 = ax1.twinx()
    plt.plot(y[:,1], 'g', lw=1.5, label='2nd')
    plt.plot(y[:, 1], 'ro')
    plt.legend(loc=0)
    plt.ylabel('value 2nd')
    # plt.grid(True)
    plt.savefig(PATH + 'two_dim_grid.png', dpi=300)
    plt.close()

def sep_plots():
    np.random.seed(2000)
    y = np.random.standard_normal((20,2)).cumsum(axis=0)
    y[:, 0] = y[:, 0] * 100
    plt.figure(figsize=(7,5))
    # subplot(211) --> 2 rowa, 1 col, this is fignumber 1
    plt.subplot(211)
    plt.plot(y[:, 0], lw=1.5, label='1st')
    plt.plot(y[:,0], 'ro')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.ylabel('value')
    plt.title('A Simple Plot')
    # subplot(212) --> 2 rowa, 1 col, this is fignumber 2
    plt.subplot(212)
    plt.plot(y[:, 1], lw=1.5, label='2nd')
    plt.plot(y[:, 1], 'ro')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.xlabel('index')
    plt.ylabel('value')
    plt.savefig(PATH + 'sep_plots.png', dpi=300)
    plt.close()

def line_bar():
    np.random.seed(2000)
    y = np.random.standard_normal((20,2)).cumsum(axis=0)
    y[:, 0] = y[:, 0] * 100
    plt.figure(figsize=(9, 4))
    plt.subplot(121)
    plt.plot(y[:, 0], lw=1.5, label='1st')
    plt.plot(y[:, 0], 'ro')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.xlabel('index')
    plt.ylabel('value')
    plt.title('1st Data Set')
    plt.subplot(122)
    plt.bar(np.arange(len(y)), y[:, 1], width=0.5, color='g', label='2nd')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.xlabel('index')
    plt.title('2nd Data Set')
    plt.savefig(PATH + 'line_bar.png', dpi=300)
    plt.close()

def scatter():
    y = np.random.standard_normal((1000, 2))
    plt.figure(figsize=(7,5))
    plt.plot(y[:, 0], y[:, 1], 'ro')
    plt.grid(True)
    plt.xlabel('1st')
    plt.ylabel('2nd')
    plt.title('Scatter Plot')
    plt.savefig(PATH + 'scatter.png', dpi=300)
    plt.close()
    
    # This time using scatter function
    plt.figure(figsize=(7,5))
    plt.scatter(y[:, 0], y[:, 1], marker='o')
    plt.grid(True)
    plt.xlabel('1st')
    plt.ylabel('2nd')
    plt.title('Scatter Plot')
    plt.savefig(PATH + 'scatter2.png', dpi=300)
    plt.close()
    
    c = np.random.randint(0, 10, len(y))
    plt.figure(figsize=(7,5))
    plt.scatter(y[:, 0], y[:, 1], c=c, marker='o')
    plt.colorbar()
    plt.grid(True)
    plt.xlabel('1st')
    plt.ylabel('2nd')
    plt.title('Scatter Plot')
    plt.savefig(PATH + 'scatter_heat.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(7,4))
    plt.hist(y, label=['1st', '2nd'], bins=25)
    plt.grid(True)
    plt.legend(loc=0)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title('Histogram')
    plt.savefig(PATH + 'hist.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(7,4))
    plt.hist(y, label=['1st', '2nd'], color=['b', 'g'], stacked=True, bins=20)
    plt.grid(True)
    plt.legend(loc=0)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title('Histogram')
    plt.savefig(PATH + 'hist_stacked.png', dpi=300)
    plt.close()
    
def box_plot():
    y = np.random.standard_normal((1000, 2))
    fig, ax = plt.subplots(figsize=(7,4))
    plt.boxplot(y)
    plt.grid(True)
    plt.setp(ax, xticklabels=['1st', '2nd'])
    plt.xlabel('data set')
    plt.ylabel('value')
    plt.title('Boxplot')
    plt.savefig(PATH + 'boxplot.png', dpi=300)
    plt.close()
    
def func(x):
    return 0.5 * np.exp(x) + 1

def exponential():
    a, b = 0.5, 1.5 # integral limits
    x = np.linspace(0, 2)
    y = func(x)
    fig, ax = plt.subplots(figsize=(7,5))
    plt.plot(x, y, 'b', linewidth=2)
    plt.ylim(ymin=0)
    # Illustrate the integral value, i.e. the area under the function
    # between the lower and upper limits
    Ix = np.linspace(a, b)
    Iy = func(Ix)
    verts = [(a, 0)] + list(zip(Ix, Iy)) + [(b, 0)]
    poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
    ax.add_patch(poly)
    plt.text(0.5 * (a+b), 1, r"$\int_a^b f(x)\mathrm{d}x$",
            horizontalalignment='center', fontsize=20)
    plt.figtext(0.9, 0.075, '$x$')
    plt.figtext(0.075, 0.9, '$f(x)$')
    
    ax.set_xticks((a,b))
    ax.set_xticklabels(('$a$', '$b$'))
    ax.set_yticks([func(a), func(b)])
    ax.set_yticklabels(('$f(a)$', '$f(b)$'))
    plt.grid(True)
    plt.savefig(PATH + 'exponential.png', dpi=300)
    plt.close()
    
def matplot_finance():
    start = (2014, 5, 1)
    end = (2014, 6, 30)
    pdb.set_trace()
    quotes = mpf.quotes_historical_yahoo_ohlc('DIA', start, end)
    fig, ax = plt.subplots(figsize=(8,5))
    fig.subplots_adjust(bottom=0.2)
    mpf.candlestick_ohlc(ax, quotes, width=0.6, colorup='b', colordown='r')
    plt.grid(True)
    ax.xaxis_date()
    ax.autoscale_view()
    # Rotate labels by 30 degrees
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.savefig(PATH + 'candle.png', dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    mpf.plot_day_summary_ohlc(ax, quotes, colorup='b', colordown='r')
    plt.grid(True)
    ax.xaxis_date()
    plt.title('DAX Index')
    plt.ylabel('index level')
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.savefig(PATH + 'plot_day_sum.png', dpi=300)
    plt.close()

    quotes = np.array(mpf.quotes_historical_yahoo_ohlc('YHOO', start, end))
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8,6))
    mpf.candlestick_ohlc(ax1, quotes, width=0.6, colorup='b', colordown='r')
    ax1.set_title('Yahoo Inc.')
    ax1.set_ylabel('Index Level')
    ax1.grid(True)
    ax1.xaxis_date()
    plt.bar(quotes[:, 0] - 0.25, quotes[:, 5], width=0.5)
    ax2.set_ylabel('volume')
    ax2.grid(True)
    ax2.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.savefig(PATH + 'candle2plot.png', dpi=300)
    plt.close()

def three_d():
    strike = np.linspace(50, 150, 24)
    ttm = np.linspace(0.5, 2.5, 24)
    strike, ttm = np.meshgrid(strike, ttm)
    iv = (strike - 100) ** 2 / (100 * strike) / ttm
    fig = plt.figure(figsize=(9,6))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(strike, ttm, iv, rstride=2, cstride=2, 
    cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)
    ax.set_xlabel('strike')
    ax.set_ylabel('time-to-maturity')
    ax.set_zlabel('implied volatility')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(PATH + 'three_d.png', dpi=300)
    plt.close()
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30,60)
    ax.scatter(strike, ttm, iv, zdir='z', s=25, c='b', marker='^')
    ax.set_xlabel('strike')
    ax.set_ylabel('time-to-maturity')
    ax.set_zlabel('implied volatility')
    plt.savefig(PATH + 'three_d2.png', dpi=300)
    plt.close()
    
if __name__ ==  "__main__":
    # matplot_finance()
    exponential()
    

