import pymysql
import datetime, sys, pdb
import pandas as pd
# from app import app
sys.path.append("/home/ec2-user/environment/python_for_finance/")
sys.path.append("/home/ec2-user/environment/python_for_finance/utils/")
from helper_funcs import stringify
from data_grab.fmp_helper import COL_MAPS

# https://docs.c9.io/docs/setup-a-database
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-LAMP.html
# mysql db pw: kmac7272

# from config import DBUSER, DBPASSWORD, DB, DBHOST
DBUSER = 'root'
DBPASSWORD = 'kmac7272'
DB = 'finance'
# DBHOST = 'mccarvik-playground-2305615'
# DBPASSWORD = 'kmac7272'
DBHOST = 'localhost'

class DBHelper:
    ''' Class to help with DB actions
        This is compatible with a mySQL DB
    '''
    def __init__(self):
        self.cnx = None
        self.cursor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cursor:
            self.cursor.close()
        if self.cnx:
            self.cnx.close()
    
    def connect(self, database=DB, db_user=DBUSER, db_password=DBPASSWORD, db_host=DBHOST):
        try:
            self.cnx = pymysql.connect(user=db_user, password=db_password, host=db_host,database=database)
            self.cursor = self.cnx.cursor()
            # app.logger.info('Successfully connected to ' + database)
        except Exception as e:
            # app.logger.info('COULD NOT CONNECT TO ' + database)
            # app.logger.info("DB ERROR:" + str(e))
            print("DB ERROR:" + str(e))
    
    def select(self, table, cols =['*'], where=None):
        cols = ",".join(cols)
        exec_string = '''SELECT {0} 
                         FROM {1} '''.format(cols,table)
        if where:
            exec_string += "WHERE {0}".format(where)
        try:
            self.cursor.execute(exec_string)
            rows = []
            headers = [h[0] for h in self.cursor.description]
            for row in self.cursor:
                rows.append(row)
            return pd.DataFrame(rows, columns=headers)
            
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("DB SELECT ERROR: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
            # app.logger.info("DB SELECT ERROR: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
            return {'status': 500}
    
    def update(self, table, cols, vals, where):
        exec_string = 'UPDATE {0}'.format(table)
        set_string = ' SET '
        vals =stringify(vals)
        for c,v in zip(cols, vals):
            set_string += '{0}={1}, '.format(c,v)
        set_string = set_string[:-2]
        exec_string += set_string
        exec_string += " WHERE {0}".format(where)
        
        try:
            self.cursor.execute(exec_string)
            self.cnx.commit()
            return {'status': 200}
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            # import pdb; pdb.set_trace()
            print("DB UPDATE ERROR: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
            # app.logger.info("DB UPDATE ERROR: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
            return {'status': 500}
    
    def insert_into(self, table, cols, vals):
        vals = stringify(vals)
        exec_string = 'INSERT INTO {0} '.format(table)
        col_string = "(" + ",".join(cols) + ")"
        val_string = "VALUES (" + ",".join(vals) + ")"
        exec_string += """
                        {0}
                        {1}""".format(col_string, val_string)
        try:
            self.cursor.execute(exec_string)
            self.cnx.commit()
            return {'status': 200}
        except pymysql.err.IntegrityError as e:
            # Duplicate entry for this insert
            return {'status': 500}
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            # import pdb; pdb.set_trace()
            print("DB INSERT INTO ERROR: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
            # app.logger.info("DB INSERT INTO ERROR: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
            return {'status': 500}
    
    def upsert(self, table, cols_vals, prim_keys):
        # first try an insert
        try:
            ret = self.insert_into(table, cols_vals.keys(), list(cols_vals.values()))
            if ret['status'] == 200:
                self.cnx.commit()
            else:
                raise Exception('Error in insert statement (Probably a duplicate')
        except:
            try:
                # if error try update (need to build where clause first)
                w_c = ""
                for pk in prim_keys:
                    w_c += pk + "=" + stringify(cols_vals[pk]) + " AND "
                w_c = w_c[:-5]
                self.update(table, cols_vals.keys(), list(cols_vals.values()), w_c)
                self.cnx.commit()
                return {'status': 200}
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print("DB UPSERT ERROR: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
                # app.logger.info("DB UPSERT ERROR: {0}, {1}, {2}".format(exc_type, exc_tb.tb_lineno, exc_obj))
                return {'status': 500}


def restart():
    import os
    os.system("sudo /etc/init.d/mysql restart")


def data_dump(table='morningstar'):
    pdb.set_trace()
    path = '/home/ubuntu/workspace/finance/app/data/' + table + ".csv"
    with DBHelper() as db:
        db.connect()
        df = db.select(table)
        df.to_csv(path)


def create_eod_px_table():
    columns_sql = {}
    columns_sql['date'] = 'date'
    columns_sql['px'] = 'float'
    columns_sql['tick'] = 'varchar(16)'
    prim_keys = ['date', 'tick']
    create_table_if_not_exists('finance', 'eod_px', columns_sql, prim_keys)


def create_fin_ratios_table():
    columns_sql = {}
    columns_sql['year'] = 'varchar(16)'
    columns_sql['month'] = 'varchar(16)'
    columns_sql['tick'] = 'varchar(16)'
    columns_sql['currentRatio'] = 'float'
    columns_sql['grossProfitMargin'] = 'float'
    columns_sql['operatingProfitMargin'] = 'float'
    columns_sql['pretaxProfitMargin'] = 'float'
    columns_sql['netProfitMargin'] = 'float'
    columns_sql['effectiveTaxRate'] = 'float'
    columns_sql['returnOnAssets'] = 'float'
    columns_sql['returnOnEquity'] = 'float'
    columns_sql['returnOnCapitalEmployed'] = 'float'
    columns_sql['nIperEBT'] = 'float'
    columns_sql['eBTperEBIT'] = 'float'
    columns_sql['eBITperRevenue'] = 'float'
    columns_sql['debtRatio'] = 'float'
    columns_sql['debtEquityRatio'] = 'float'
    columns_sql['longtermDebtToCapitalization'] = 'float'
    columns_sql['totalDebtToCapitalization'] = 'float'
    columns_sql['interestCoverageRatio'] = 'float'
    columns_sql['cashFlowToDebtRatio'] = 'float'
    columns_sql['companyEquityMultiplier'] = 'float'
    columns_sql['fixedAssetTurnover'] = 'float'
    columns_sql['assetTurnover'] = 'float'
    columns_sql['operatingCashFlowSalesRatio'] = 'float'
    columns_sql['freeCashFlowOperatingCashFlowRatio'] = 'float'
    columns_sql['cashFlowCoverageRatios'] = 'float'
    columns_sql['capitalExpenditureCoverageRatios'] = 'float'
    columns_sql['shortTermCoverageRatios'] = 'float'
    columns_sql['dividendPayoutRatio'] = 'float'
    columns_sql['priceBookValueRatio'] = 'float'
    columns_sql['priceCashFlowRatio'] = 'float'
    columns_sql['priceEarningsRatio'] = 'float'
    columns_sql['priceEarningsToGrowthRatio'] = 'float'
    columns_sql['priceSalesRatio'] = 'float'
    columns_sql['dividendYield'] = 'float'
    columns_sql['enterpriseValueMultiple'] = 'float'
    columns_sql['priceFairValue'] = 'float'
    columns_sql['dividendpaidAndCapexCoverageRatios'] = 'float'
    prim_keys = ['year', 'month', 'tick']
    create_table_if_not_exists('finance', 'fin_ratios', columns_sql, prim_keys)


def create_bal_sheet_table():
    columns_sql = {}
    columns_sql['year'] = 'varchar(16)'
    columns_sql['month'] = 'varchar(16)'
    columns_sql['tick'] = 'varchar(16)'
    for val in COL_MAPS['bal_sheet'].values():
        columns_sql[val] = 'float'
    prim_keys = ['year', 'month', 'tick']
    create_table_if_not_exists('finance', 'bal_sheet', columns_sql, prim_keys)

def create_inc_statement_table():
    columns_sql = {}
    columns_sql['year'] = 'varchar(16)'
    columns_sql['month'] = 'varchar(16)'
    columns_sql['tick'] = 'varchar(16)'
    for val in COL_MAPS['inc_statement'].values():
        columns_sql[val] = 'float'
    prim_keys = ['year', 'month', 'tick']
    create_table_if_not_exists('finance', 'inc_statement', columns_sql, prim_keys)


def create_table_if_not_exists(schema, table, columns_dict, prim_keys):
    columns_list = []
    for name, dtype in columns_dict.items():
        columns_list.append(name + ' ' + dtype)
    columns_sql = ', '.join(columns_list)
    prim_keys_sql = ', '.join(prim_keys)
    create_sql = "create table if not exists {}.{} ({})".format(schema, table, columns_sql)
    prim_keys_sql = "alter table {}.{} add primary key ({})".format(schema, table, prim_keys_sql)
    with DBHelper() as dbh:
        dbh.connect()
        dbh.cursor.execute(create_sql)
        dbh.cursor.execute(prim_keys_sql)
    

if __name__ == '__main__':
    # a = DBHelper()
    # a.connect(db_host='localhost')
    # restart()
    try:
        create_inc_statement_table()
    except Exception as e:
        pdb.set_trace()
        print()
        print(e)