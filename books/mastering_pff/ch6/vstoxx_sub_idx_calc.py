import calendar as cal
import datetime as dt
import pandas as pd
import math, pdb
from dateutil.relativedelta import relativedelta
import numpy as np
import urllib.request
from lxml import html
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

data_file = "/home/ubuntu/workspace/python_for_finance/mastering_pff/ch6/vsi.csv"
vstoxx_file = "/home/ubuntu/workspace/python_for_finance/mastering_pff/ch6/vstoxx.csv"
IMG_PATH = "/home/ubuntu/workspace/python_for_finance/mastering_pff/png/"

class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

OPENER = AppURLopener()


def calculate_vstoxx_index(dataframe, col_name):    
    secs_per_day = float(60*60*24)
    utility = OptionUtility()
    
    for row_date, row in dataframe.iterrows():
        # Set each expiry date with an 
        # expiration time of 5p.m
        date = row_date.replace(hour=17)  
        
        # Ensure dates and sigmas are in legal range
        expiry_date_1 = utility.get_settlement_date(date)
        expiry_date_2 = utility.fwd_expiry_date(date, 1)
        days_diff = (expiry_date_1-date).days
        sigma_1, sigma_2 = row["V6I1"], row["V6I2"]        
        if -1 <= days_diff <= 1:
            sigma_1, sigma_2 = row["V6I2"], row["V6I3"]        
        if days_diff <= 1:
            expiry_date_1 = expiry_date_2
            expiry_date_2 = utility.fwd_expiry_date(date, 2)   
            
        # Get expiration times in terms of seconds
        Nti = (expiry_date_1-date).total_seconds()
        Nti1 = (expiry_date_2-date).total_seconds()
        
        # Calculate index as per VSTOXX formula in seconds
        # This is the formula used by VSTOXX for their main index calculation
        first_term = \
            (Nti1-30*secs_per_day)/ \
            (Nti1-Nti)*(sigma_1**2)*Nti/ \
            (secs_per_day*365)
        second_term = \
            (30*secs_per_day-Nti)/ \
            (Nti1-Nti)*(sigma_2**2)*Nti1/ \
            (secs_per_day*365)
        sub_index = math.sqrt(365.*(first_term+second_term)/30.)    
        dataframe.set_value(row_date, col_name, sub_index)
    return dataframe


class OptionUtility(object):
    def get_settlement_date(self, date):
        """ Get third friday of the month """
        day = 21 - (cal.weekday(date.year, date.month, 1) + 2) % 7
        return dt.datetime(date.year, date.month, day, 12, 0, 0)

    def get_date(self, web_date_string, date_format):
        """  Parse a date from the web to a date object """
        return dt.datetime.strptime(web_date_string, date_format)

    def fwd_expiry_date(self, current_dt, months_fws):
        return self.get_settlement_date(
            current_dt + relativedelta(months=+months_fws))


class VSTOXXCalculator(object):
    def __init__(self):
        self.secs_per_day = float(60*60*24)
        self.secs_per_year = float(365*self.secs_per_day)

    def calculate_sub_index(self, df, t_calc, t_settle, r):
        # Actual calculation of sub_index
        T = (t_settle-t_calc).total_seconds()/self.secs_per_year
        R = math.exp(r*T)

        # Calculate dK
        df["dK"] = 0
        df["dK"][df.index[0]] = df.index[1]-df.index[0]
        df["dK"][df.index[-1]] = df.index[-1]-df.index[-2]
        df["dK"][df.index[1:-1]] = (df.index.values[2:]-
                                    df.index.values[:-2])/2
        # Calculate the forward price
        df["AbsDiffCP"] = abs(df["Call"]-df["Put"])
        min_val = min(df["AbsDiffCP"])
        f_df = df[df["AbsDiffCP"]==min_val]
        fwd_prices = f_df.index+R*(f_df["Call"]-f_df["Put"])
        F = np.mean(fwd_prices)

        # Get the strike not exceeding forward price
        K_i0 = df.index[df.index <= F][-1]

        # Calculate M(K(i,j))
        df["MK"] = 0
        df["MK"][df.index < K_i0] = df["Put"]
        df["MK"][K_i0] = (df["Call"][K_i0]+df["Put"][K_i0])/2.
        df["MK"][df.index > K_i0] = df["Call"]

        # Apply the variance formula to get the sub-index
        # This is the formula used by VSTOXX to calculate the index
        summation = sum(df["dK"]/(df.index.values**2)*R*df["MK"])
        variance = 2/T*summation-1/T*(F/float(K_i0)-1)**2
        subindex = 100*math.sqrt(variance)
        return subindex


class EurexWebPage(object):
    def __init__(self):
        self.url = "%s%s%s%s%s" % (
            "http://www.eurexchange.com/",
            "exchange-en/market-data/statistics/",
            "market-statistics-online/180102!",
            "onlineStats?productGroupId=846&productId=19068",
            "&viewType=3")
        self.param_url = "&cp=%s&month=%s&year=%s&busDate=%s"
        self.lastupdated_dateformat = "%b %d, %Y %H:%M:%S"
        self.web_date_format = "%Y%m%d"
        self.__strike_price_header__ = "Strike price"
        self.__prices_header__ = "Daily settlem. price"
        self.utility = OptionUtility()

    def get_available_dates(self):
        html_data = OPENER.open(self.url).read()
        # html_data = urlopen(req).read()
        webpage = html.fromstring(html_data)

        # Find the dates available on the website
        dates_listed = webpage.xpath("//select[@name='busDate']" + "/option")
        return [date_element.get("value")for date_element in reversed(dates_listed)]

    def get_date_from_web_date(self, web_date):
        return self.utility.get_date(web_date, self.web_date_format)

    def get_option_series_data(self, is_call, current_dt, option_dt):
        selected_date = current_dt.strftime(self.web_date_format)
        option_type = "Call" if is_call else "Put"
        target_url = (self.url + self.param_url) % (option_type,
                                         option_dt.month,
                                         option_dt.year,
                                         selected_date)
        html_data = OPENER.open(target_url).read()
        webpage = html.fromstring(html_data)
        update_date = self.get_last_update_date(webpage)
        indexes = self.get_data_headers_indexes(webpage)
        data = self.__get_data_rows__(webpage, indexes, option_type)
        return data, update_date

    def __get_data_rows__(self, webpage, indexes, header):
        data = pd.DataFrame()
        for row in webpage.xpath("//table[@class='dataTable']/" + "tbody/tr"):
            columns = row.xpath("./td")
            if len(columns) > max(indexes):
                try:
                    [K, price] = [float(columns[i].text.replace(",","")) for i in indexes]
                    data.set_value(K, header, price)
                except:
                    continue
        return data

    def get_data_headers_indexes(self, webpage):
        table_headers = webpage.xpath("//table[@class='dataTable']" + "/thead/th/text()")
        indexes_of_interest = [
            table_headers.index(self.__strike_price_header__),
            table_headers.index(self.__prices_header__)]
        return indexes_of_interest

    def get_last_update_date(self, webpage):
        return dt.datetime.strptime(webpage.xpath("//p[@class='date']/b")[-1].text, self.lastupdated_dateformat)


class VSTOXXSubIndex:
    def __init__(self, path_to_subindexes):
        self.sub_index_store_path = path_to_subindexes
        self.utility = OptionUtility()
        self.webpage = EurexWebPage()
        self.calculator = VSTOXXCalculator()
        self.csv_date_format = "%m/%d/%Y"

    def start(self, months=2, r=0.015):
        # For each date available, fetch the data
        for selected_date in self.webpage.get_available_dates():
            print("Collecting historical data for %s..." % selected_date)
            try:
                self.calculate_and_save_sub_indexes(selected_date, months, r)
            except Exception as e:
                continue
        print("Completed.")
    
    def calculate_and_save_sub_indexes(self, selected_date, months_fwd, r):
        current_dt = self.webpage.get_date_from_web_date(selected_date)
        for i in range(1, months_fwd+1):
            # Get settlement date of the expiring month
            expiry_dt = self.utility.fwd_expiry_date(current_dt, i)
                
            # Get calls and puts of expiring month
            dataset, update_dt = self.get_data(current_dt, expiry_dt)                        
            if not dataset.empty:
                sub_index = self.calculator.calculate_sub_index(dataset, update_dt, expiry_dt, r)
                self.save_vstoxx_sub_index_to_csv(current_dt, sub_index, i)

    def save_vstoxx_sub_index_to_csv(self, current_dt, sub_index, month):
        subindex_df = None
        try:
            subindex_df = pd.read_csv(self.sub_index_store_path, index_col=[0])
        except:
            subindex_df = pd.DataFrame()
            
        display_date = current_dt.strftime(self.csv_date_format)
        subindex_df.set_value(display_date, "I" + str(month), sub_index)
        subindex_df.to_csv(self.sub_index_store_path)

    def get_data(self, current_dt, expiry_dt):
        """ Fetch and join calls and puts option series data """
        calls, dt1 = self.webpage.get_option_series_data(True, current_dt, expiry_dt)
        puts, dt2 = self.webpage.get_option_series_data(False, current_dt, expiry_dt)
        option_series = calls.join(puts, how='inner')            
        if dt1 != dt2:           
            print("Error: 2 different underlying prices.")
        return option_series, dt1

if __name__ == '__main__':
    # run calc and save down data
    # vsi = VSTOXXSubIndex(data_file)
    # vsi.start()
    
    vstoxx_sub_indexes = pd.read_csv(data_file, index_col=[0], parse_dates=True, dayfirst=False)
    vstoxx = pd.read_csv(vstoxx_file, index_col=[0], parse_dates=True, dayfirst=False)
    
    # DONT Have the data for the website to overlap dates
    # start_dt = min(vstoxx_sub_indexes.index.values)
    # vstoxx = vstoxx[vstoxx.index >= start_dt]
    
    # new_pd = pd.DataFrame(vstoxx_sub_indexes["I2"])
    # new_pd = new_pd.join(vstoxx["V6I2"], how='inner')
    # new_pd.plot(figsize=(10, 6), grid=True)
    # plt.savefig(IMG_PATH + 'vstoxx_sub_comp.png', dpi=300)
    # plt.close()
    
    pdb.set_trace()
    # comparing our calculation to the main VSTOXX index
    sample = vstoxx.tail(100)  # From the previous section 
    sample = calculate_vstoxx_index(sample, "Calculated")

    vstoxx_df = sample["V2TX"]
    calculated_df = sample["Calculated"]
    df = pd.DataFrame({'VSTOXX' : sample["V2TX"],'Calculated' : sample["Calculated"]})
    # df.plot(figsize=(10, 6), grid=True, style=['ro','b'])
    # plt.savefig(IMG_PATH + 'vstoxx_main_comp.png', dpi=300)
    # plt.close()