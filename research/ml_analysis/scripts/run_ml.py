"""
Main functon to kick off the machine learning analysis
"""
import pdb
import time
import sys
# import warnings
# import datetime as dt
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("/home/ec2-user/environment/python_for_finance/")
from utils.helper_funcs import timeme
from utils.db_utils import DBHelper
from utils.ml_utils import standardize
from utils.data_utils import DAY_COUNTS, PER_SHARE, RETURNS, FWD_RETURNS, \
                             MARGINS, INDEX, RATIOS, OTHER

# from ml_algorithms import *
# from scripts.model_evaluation import *
# from scripts.feature_selection import *
# from scripts.ensemble_methods import *
# from scripts.continuous_variables import *

FILE_PATH = '/home/ec2-user/environment/python_for_finance/data_grab/'
FILE_NAME = 'fmp_available_stocks_20190507.txt'


def run(inputs, label='retfwd_2y', cust_ticks=None):
    """
    Main function to run analytics
    """
    time0 = time.time()
    if cust_ticks:
        tickers = cust_ticks
    else:
        tickers = list(pd.read_csv(FILE_PATH + FILE_NAME, header=None)[0].values)

    with DBHelper() as dbh:
        dbh.connect()
        lis = ''
        for tick in tickers:
            lis += "'" + tick + "', "
        print("starting data retrieval")
        df_ret = dbh.select('fin_ratios', where='tick in (' + lis[:-2] + ')'
                            'and {} != 0'.format(label))

    # Getting Dataframe
    time1 = time.time()
    print("Done Retrieving data, took {0} seconds".format(time1-time0))

    # grab the more recent data for testing later
    # these wont have a target becuase the data is too recent
    test_df, train_df = separate_train_test(df_ret)
    filtered_test_df = filter_live(test_df)

    # clean data
    train_df = remove_unnecessary_columns(train_df)
    print("Number of rows in training set: {}".format(len(train_df)))
    train_df = clean_data(train_df)
    print("Number of rows in training set after cleaning: {}"
          "".format(len(train_df)))
    pdb.set_trace()
    train_df = add_target(train_df, label, breaks=1, custom_breaks=[33, 67])
    train_df = train_df.set_index(['ticker', 'date'])
    train_df = selectInputs(train_df, inputs)
    # drop all rows with NA's
    size_before = len(train_df)
    train_df = train_df.dropna()
    print("There are {0} samples (removed {1} NA rows)".format(len(train_df), size_before - len(train_df)))
    pdb.set_trace()

    # Select features
    feature_selection(train_df, inputs)

    # Feature Extraction
    # feature_extraction(df, inputs)

    # Algorithms
    # timeme(run_perceptron)(df, tuple(inputs))
    # timeme(adalineGD)(df, tuple(inputs))
    # timeme(adalineSGD)(df, tuple(inputs))
    # timeme(run_perceptron_multi)(df, tuple(inputs))
    # model = timeme(logisticRegression)(df, tuple(inputs), C=100, penalty='l2')

    # model = timeme(k_nearest_neighbors)(df, tuple(inputs), k=8)
    model = timeme(random_forest)(df, tuple(inputs), estimators=3)
    # model = timeme(support_vector_machines)(df, tuple(inputs), C=100)
    # timeme(nonlinear_svm)(df, tuple(inputs), C=1)
    # timeme(decision_tree)(df, tuple(inputs), md=4)
    # timeme(adalinesgd)(df, tuple(inputs), estimators=3)
    # timeme(run_perceptron_multi)(df, tuple(inputs), estimators=3)


    # Model Evaluation
    # model_evaluation(df, inputs)
    # timeme(majority_vote)(df, tuple(inputs))
    # timeme(bagging)(df, tuple(inputs))
    # timeme(adaboost)(df, tuple(inputs))
    # timeme(heat_map)(df, tuple(inputs))
    # timeme(linear_regressor)(df, tuple(inputs))
    # timeme(linear_regression_sklearn)(df, tuple(inputs))
    # timeme(ransac)(df, tuple(inputs))
    # timeme(polynomial_regression)(df, tuple(inputs))
    # timeme(nonlinear)(df, tuple(inputs))
    # timeme(random_forest_regression)(df, tuple(inputs))

    # test on recent data
    preds = evalOnCurrentCompanies(model, filtered_cur_df, inputs)
    pdb.set_trace()
    print()


def evalOnCurrentCompanies(model, df, inputs):
    pdb.set_trace()
    df_ind = df[['ticker', 'date', 'month']]
    df_trimmed = pd.DataFrame(standardize(df[inputs]), columns=inputs)
    df_combine = pd.concat([df_ind.reset_index(drop=True), df_trimmed], axis=1)
    predictions = {}
    for ix, row in df_combine.iterrows():
        print(row['ticker'] + "   " + row['date'] + "   " + str(row['month']), end="")
        pred = model.predict(row[inputs])[0]
        try:
            predictions[pred].append(row['ticker'])
        except:
            predictions[pred] = [row['ticker']]
        print("    Class Prediction: " + str(pred))
    return predictions


def feature_selection(train_df, inputs):
    """
    Sequential Backward Selection - feature selection to see
    which are the most telling variable
    Default is K-means Clustering
    """
    timeme(sbs_run)(train_df, tuple(inputs))
    # timeme(sbs_run)(train_df, tuple(inputs), est=DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0))
    # timeme(sbs_run)(train_df, tuple(inputs), est=RandomForestClassifier(criterion='entropy', n_estimators=3, random_state=1,n_jobs=3))
    # timeme(sbs_run)(train_df, tuple(inputs), est=SVC(kernel='linear', C=100, random_state=0))
    # timeme(sbs_run)(train_df, tuple(inputs), est=LogisticRegression(C=100, random_state=0, penalty='l1'))
    # timeme(sbs_run)(train_df, tuple(inputs), est=AdalineSGD(n_iter=15, eta=0.001, random_state=1))
    # timeme(sbs_run)(train_df, tuple(inputs), est=AdalineGD(n_iter=20, eta=0.001))

    # Random Forest Feature Selection - using a random forest to identify
    # which factors decrease impurity the most
    timeme(random_forest_feature_importance)(train_df, tuple(inputs))

    # Logistic Regression Feature Selection - logistic regression
    # should expose the important variables through its weights
    timeme(logistic_regression_feature_importance)(train_df, tuple(inputs))


def feature_extraction(df, inputs):
    # Transforms the data - can be used to linearly separate data thru dimensionality reduction
    timeme(principal_component_analysis)(df, tuple(inputs))
    # timeme(pca_scikit)(df, tuple(inputs))
    # timeme(linear_discriminant_analysis)(df, tuple(inputs))
    # timeme(lda_scikit)(df, tuple(inputs))


def model_evaluation(df, inputs):
    timeme(kfold_cross_validation)(df, tuple(inputs))
    # timeme(learning_curves)(df, tuple(inputs))
    # timeme(validation_curves)(df, tuple(inputs))
    # timeme(grid_search_analysis)(df, tuple(inputs))
    # timeme(precision_vs_recall)(df, tuple(inputs))


def separate_train_test(data):
    """
    Separate the training and testing data
    """
    test_df = data[data.year == '2018']
    train_df = data[data.year != '2018']
    return test_df, train_df


def selectInputs(df, inputs):
    columns = inputs + ['target'] + ['target_proxy']
    df = df[columns]
    return df


def add_target(data_df, tgt, breaks=2, custom_breaks=None):
    """
    Add the target column to be the label
    """
    num_of_breaks = breaks
    data_df['target_proxy'] = data_df[tgt]
    data_df = data_df.dropna(subset=['target_proxy'])
    data_df = data_df[data_df['target_proxy'] != 0]

    if not custom_breaks:
        break_arr = np.linspace(0, 100, num_of_breaks+1)[1:-1]
    else:
        break_arr = custom_breaks
    breaks = np.percentile(data_df['target_proxy'], break_arr)
    # breaks = np.percentile(data_df['target_proxy'], [50])
    data_df['target'] = data_df.apply(lambda x: 
                                      targetToCatMulti(x['target_proxy'], breaks), axis=1)
    return data_df


def targetToCatMulti(x, breaks):
    cat = 0
    for b in breaks:
        if x < b:
            return cat
        cat += 1
    return cat


def remove_unnecessary_columns(data_df):
    """
    Filter to only the columns we want
    """
    # data_df = data_df[RATIOS + KEY_STATS + OTHER + GROWTH + MARGINS + RETURNS +
    #              FWD_RETURNS + PER_SHARE + INDEX]
    data_df = data_df[PER_SHARE + DAY_COUNTS + RETURNS + FWD_RETURNS + INDEX + 
                      MARGINS + RATIOS + OTHER]
    return data_df


def filter_live(test_df):
    """
    Filter the live options by some boundary constraints
    """
    test_df = test_df[test_df['pe_ratio'] < 100]
    test_df = test_df[test_df['pe_ratio'] > 0]
    test_df = test_df[test_df['pb_ratio'] < 10]
    test_df = test_df[test_df['ps_ratio'] < 10]
    test_df = test_df[test_df['capex_to_rev'] < 100]
    return test_df


def clean_data(train_df):
    """
    Remove any outlier data
    """
    # To filter out errant data
    train_df = train_df[train_df['pe_ratio'] != 0]
    train_df = train_df[train_df['pb_ratio'] > 0]

    # To filter out outliers
    # train_df = train_df[train_df['capExToSales'] < 20]
    # train_df = train_df[abs(train_df['revenueGrowth']) < 200]
    # train_df = train_df[train_df['trailingPE'] > 0]
    # train_df = train_df[abs(train_df['sharpeRatio']) < 7]
    # train_df = train_df[train_df['sharpeRatio'] > 0]

    # Custom Filters for pruning
    # train_df = train_df[abs(train_df['trailingPE']) < 30]
    # train_df = train_df[abs(train_df['priceToBook']) < 10]
    # train_df = train_df[train_df['divYield'] > 0]
    # train_df = train_df[train_df['divYield'] < 8]
    # train_df = train_df[train_df['debtToEquity'] < 10]
    # train_df = train_df[train_df['returnOnEquity'] > 0]
    # train_df = train_df[train_df['returnOnEquity'] < 50]
    # train_df = train_df[train_df['currentRatio'] < 10]


    # only look at the top and bottom percentile ranges
    # train_df = train_df[(train_df['target'] == 0) | (train_df['target'] == 4)]
    return train_df


if __name__ == "__main__":
    # Most Relevant columns
    COLS = ['roe', 'pb_ratio', 'div_yield', 'price_to_sales', 'pe_ratio']
    # TICKS = ['A', 'AAPL']
    run(COLS)
    