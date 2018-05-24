import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

COLORS = ["b", "c", "m", "g", "k", "r", "y", "w"]
MARKERS = ['-', '--', '-.', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2',
            '3', '4', '5', 's', 'p', '*', 'h', 'H', 'D', 'd', '|', 'x', '+', 'o']

def autolabel(rects, ax, decs=2):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                round(height,decs),
                ha='center', va='bottom')

def format_dates(ax, fmt='%Y-%m-%d', dates=None):
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    # yearsFmt = mdates.DateFormatter('%Y')
    yearsFmt = mdates.DateFormatter(fmt)
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    if dates:
        ax.set_xlim(dates[0], dates[-1])
    ax.format_xdata = mdates.DateFormatter('%Y-%m')
    
    return ax

    