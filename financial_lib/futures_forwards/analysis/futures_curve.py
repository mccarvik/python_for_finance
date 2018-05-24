import sys
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pdb, datetime, re
from dateutil.relativedelta import relativedelta
from utils import mpl_utils
from financial_lib.data_grab.quandl_api_helper import quandl_api_dict, URL, MONTH_MAP
from financial_lib.data_grab.quandl_api import callQuandlAPI
from financial_lib.fin_lib_utils import IMG_PATH


def getFuturesCurve(fut, dt=None, end=None):
    # urls = getURLs(fut, dt)
    urls = getURLs(fut, end=end)
    data = []
    for u in urls:
        try:
            cqa = callQuandlAPI(u[0])
            data.append([cqa['Settle'].iloc[0], u[1]])
            # data.append([callQuandlAPI(u[0])['Settle'].iloc[0], u[1]])
        except:
            # pdb.set_trace()
            pat = r'.*?datasets(.*)data.*'
            match = re.search(pat, u[0])
            regex = match.group(1)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("No contract for this month and fut: {3}  {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj, regex))
    data = [[datetime.date(int(d[1][:4]), int(d[1][4:]), 1) for d in data], [float(d[0]) for d in data]]
    return data
    
    
def buildURL(db_code, ds_code, start=None, end=None):
    if not end:
        end = datetime.date.today()
    if not start:
        start = (end - datetime.timedelta(7))
    return URL.format(db_code, ds_code, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))


def getURLs(fut, today=None, end=None):
    if not end:
        end = datetime.date.today() + datetime.timedelta(365)
    urls = []
    date = datetime.date.today()
    while date < end:
        d_str = dateToStringFormat(date)
        db_code = quandl_api_dict[fut][0]
        ds_code = quandl_api_dict[fut][1] + d_str
        urls.append([buildURL(db_code, quandl_api_dict[fut][1] + d_str), str(date.year) + str(date.month)])
        one_mon_rel = relativedelta(months=1)
        date = date + one_mon_rel
    return urls

    
def dateToStringFormat(d):
    return str(MONTH_MAP[d.month]) + str(d.year)


def chartCurveSameAxis(data):
    plt.figure(figsize=(7,4))
    fig, ax1 = plt.subplots()
    col_ct = 0
    for d in data:
        ax1.plot(d[0][0], d[0][1], mpl_utils.COLORS[col_ct], lw=1.5, label=d[1], marker='o')
        col_ct+=1
    ax1.axis('tight')
    ax1.set_xlabel('Date')
    ax1 = mpl_utils.format_dates(ax1, '%Y-%m')
    
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend(loc=0)
    
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    # Set the limits of th y axis
    plt.ylim(plt.ylim()[0]*0.96, plt.ylim()[-1]*1.04)
    plt.title('Commodity Curves')
    plt.savefig(IMG_PATH + 'futures/comm_curves', dpi=300)
    plt.close()


def chartCurveDiffAxis(data):
    plt.figure(figsize=(7,4))
    fig, ax1 = plt.subplots()
    pdb.set_trace()
    ax1.plot(data[0][0][0], data[0][0][1], mpl_utils.COLORS[0], lw=1.5, label=data[0][1])
    ax1.axis('tight')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(data[0][1])
    ax1 = mpl_utils.format_dates(ax1, '%Y-%m')
    
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(data[1][0][0], data[1][0][1], mpl_utils.COLORS[1], lw=1.5, label=data[1][1])
    ax2.set_ylabel(data[1][1])
    ax1.legend(loc=0)
    
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    # Set the limits of th y axis
    plt.title('Commodity Curves')
    plt.savefig(IMG_PATH + 'comm_curves', dpi=300)
    plt.close()


if __name__ == '__main__':
    end = datetime.date.today() + datetime.timedelta(1095)
    fut = ['corn', 'gold', 'wti_oil', 'brent_oil', 'sugar']
    fut = ['wti_oil', 'brent_oil']
    curves = []
    for f in fut:
        curves.append([getFuturesCurve(f, end=end), f])
    chartCurveSameAxis(curves)
    
    
    