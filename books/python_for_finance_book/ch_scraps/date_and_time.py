import sys, pdb
sys.path.append('/usr/share/doc')
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
mpl.rcParams['font.family'] = 'serif'
import datetime as dt
import matplotlib.pyplot as plt
import pytz

PATH = '/home/ubuntu/workspace/python_for_finance/png/book_examples/dnt/'

def python_func():
    print(dt.datetime.now())
    to = dt.datetime.today()
    print(to)
    print(type(to))
    print(dt.datetime.today().weekday())
    d = dt.datetime(2016, 10, 31, 10, 5, 30, 500000)
    print(d)
    print(str(d))
    print(d.year, d.month, d.day, d.hour)
    o = d.toordinal()
    print(o)
    print(dt.datetime.fromordinal(o))
    t = dt.datetime.time(d)
    print(t)
    print(type(t))
    dd = dt.datetime.date(d)
    print(dd)
    print(d.replace(second=0, microsecond=0))
    td = d - dt.datetime.now()
    print(td)
    print(type(td))
    td.days
    print(td.seconds, td.microseconds, td.total_seconds(), d.isoformat())
    print(d.strftime("%A, %d. %B %Y %I:%M%p"))
    print(dt.datetime.strptime('2017-03-31', '%Y-%m-%d'))
    # year first and four digit year
    print(dt.datetime.strptime('30-4-16', '%d-%m-%y'))
    # day first and two digit year
    ds = str(d)
    print(ds)
    print(dt.datetime.strptime(ds, '%Y-%m-%d %H:%M:%S.%f'))
    print(dt.datetime.now())
    print(dt.datetime.utcnow())
    #  Universal Time, Coordinated
    print(dt.datetime.now() - dt.datetime.utcnow())

    u = dt.datetime.utcnow()
    u = u.replace(tzinfo=UTC())
    # attach time zone information
    print(u)
    print(u.astimezone(CET()))
    print(pytz.country_names['US'])
    print(pytz.country_timezones['BE'])
    print(pytz.common_timezones[-10:])

    u = dt.datetime.utcnow()
    u = u.replace(tzinfo=pytz.utc)
    print(u)
    print(u.astimezone(pytz.timezone("CET")))
    print(u.astimezone(pytz.timezone("GMT")))
    print(u.astimezone(pytz.timezone("US/Central")))

class CET(dt.tzinfo):
    def utcoffset(self, d):
        return dt.timedelta(hours=2)
    def dst(self, d):
        return dt.timedelta(hours=1)
    def tzname(self, d):
        return "CET + 1"
    
# UTC + 2h = CET (summer)
class UTC(dt.tzinfo):
    def utcoffset(self, d):
        return dt.timedelta(hours=0)
    def dst(self, d):
        return dt.timedelta(hours=0)
    def tzname(self, d):
        return "UTC"

def numpy_func():
    d = dt.datetime(2016, 10, 31, 10, 5, 30, 500000)
    nd = np.datetime64('2015-10-31')
    print(nd)
    print(np.datetime_as_string(nd))
    print(np.datetime_data(nd))
    print(d)
    nd = np.datetime64(d)
    print(nd)
    print(nd.astype(dt.datetime))
    nd = np.datetime64('2015-10', 'D')
    print(nd)
    print(np.datetime64('2015-10') == np.datetime64('2015-10-01'))
    print(np.array(['2016-06-10', '2016-07-10', '2016-08-10'], dtype='datetime64'))
    print(np.array(['2016-06-10T12:00:00', '2016-07-10T12:00:00',
              '2016-08-10T12:00:00'], dtype='datetime64[s]'))
    print(np.arange('2016-01-01', '2016-01-04', dtype='datetime64'))
    # daily frequency as default in this case
    print(np.arange('2016-01-01', '2016-10-01', dtype='datetime64[M]'))
    # monthly frequency
    print(np.arange('2016-01-01', '2016-10-01', dtype='datetime64[W]')[:10])
    # weekly frequency
    dtl = np.arange('2016-01-01T00:00:00', '2016-01-02T00:00:00',
                    dtype='datetime64[h]')
    # hourly frequency
    print(dtl[:10])
    
    np.random.seed(3000)
    rnd = np.random.standard_normal(len(dtl)).cumsum() ** 2
    fig = plt.figure()
    plt.plot(dtl.astype(dt.datetime), rnd)
    # convert np.datetime to datetime.datetime
    plt.grid(True)
    fig.autofmt_xdate()
    # auto formatting of datetime xticks
    # tag: datetime_plot
    # title: Plot with datetime.datetime xticks auto-formatted
    plt.savefig(PATH + 'dnt1.png', dpi=300)
    plt.close()

    print(np.arange('2016-01-01T00:00:00', '2016-01-02T00:00:00',
              dtype='datetime64[s]')[:10])
    # seconds as frequency
    print(np.arange('2016-01-01T00:00:00', '2016-01-02T00:00:00',
            dtype='datetime64[ms]')[:10])
    # milliseconds as frequency

def pandas_func():
    nd = np.datetime64('2015-10-31')
    ts = pd.Timestamp('2016-06-30')
    print(ts)
    d = ts.to_pydatetime()
    print(d)
    print(pd.Timestamp(d))
    print(pd.Timestamp(nd))
    dti = pd.date_range('2016/01/01', freq='M', periods=12)
    print(dti)
    print(dti[6])
    pdi = dti.to_pydatetime()
    print(pdi)
    print(pd.DatetimeIndex(pdi))
    print(pd.DatetimeIndex(dti.astype(pd.datetime)))
    rnd = np.random.standard_normal(len(dti)).cumsum() ** 2
    df = pd.DataFrame(rnd, columns=['data'], index=dti)
    df.plot()
    # tag: pandas_plot
    # title: Pandas plot with Timestamp xticks auto-formatted
    plt.savefig(PATH + 'dnt2.png', dpi=300)
    plt.close()

    print(pd.date_range('2016/01/01', freq='M', periods=12, tz=pytz.timezone('CET')))
    dti = pd.date_range('2016/01/01', freq='M', periods=12, tz='US/Eastern')
    print(dti)
    print(dti.tz_convert('GMT'))

if __name__ == '__main__':
    # python_func()
    # numpy_func()
    pandas_func()