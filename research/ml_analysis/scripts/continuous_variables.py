import sys, datetime, pdb, time
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor

from utils.ml_utils import plot_decision_regions, standardize, IMG_PATH, lin_regplot
from algorithms.linear_regression_gd import LinearRegressionGD



def heat_map(df, xcols):
    y = df['target']
    X = df[list(xcols)]
    cols = ['target_proxy'] + list(xcols)
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    sns.set(style='whitegrid', context='notebook')    
    sns.pairplot(df[cols], size=2.5)    
    plt.tight_layout()    
    plt.savefig(IMG_PATH + 'corr_mat.png', dpi=300)
    plt.close()
    
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, 
                cbar=True,
                annot=True, 
                square=True,
                fmt='.2f',
                annot_kws={'size': 15},
                yticklabels=cols,
                xticklabels=cols)
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'heat_map.png', dpi=300)
    plt.close()
    
def linear_regressor(df, xcols):
    y = df['target_proxy']
    X = df[list(xcols)[0]]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    lr = LinearRegressionGD()
    lr.fit(np.transpose(np.array([X_train])), y_train)
    plt.plot(range(1, lr.n_iter+1), lr.cost_)
    plt.ylabel('SSE')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'cost.png', dpi=300)
    plt.close()
    
    lin_regplot(np.transpose(np.array([X_train])), y_train, lr)
    plt.savefig(IMG_PATH + 'lin_reg_cost.png', dpi=300)
    plt.close()
    
    # Find the average return of a stock with PE = 20
    # Note: will give odd results if x values are standardized and input is not
    y_val_std = lr.predict([20.0])
    print("Estimated Return: %.3f" % y_val_std)
    print('Slope: %.3f' % lr.w_[1])
    print('Intercept: %.3f' % lr.w_[0])

def linear_regression_sklearn(df, xcols):
    y = df['target_proxy']
    X = df[list(xcols)[0]]
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    X = np.transpose(np.array([X]))      
    slr = LinearRegression()
    slr.fit(X, y.values)
    y_pred = slr.predict(X)
    print('Slope: %.3f' % slr.coef_[0])
    print('Intercept: %.3f' % slr.intercept_)
    
    lin_regplot(X, y.values, slr)
    plt.xlabel('x val')
    plt.ylabel('Return')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'scikit_lr_fit.png', dpi=300)
    plt.close()

    # Closed-form solution
    Xb = np.hstack((np.ones((X.shape[0], 1)), X))
    w = np.zeros(X.shape[1])
    z = np.linalg.inv(np.dot(Xb.T, Xb))
    w = np.dot(z, np.dot(Xb.T, y))
    print('Slope: %.3f' % w[1])
    print('Intercept: %.3f' % w[0])
    
def ransac(df, xcols):
    # function to deal with outliers
    y = df['target_proxy']
    X = df[list(xcols)[0]]
    X = np.transpose(np.array([X]))
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
         
    ransac = RANSACRegressor(LinearRegression(), 
                             max_trials=100, 
                             min_samples=50, 
                             residual_metric=lambda x: np.sum(np.abs(x), axis=1), 
                             residual_threshold=5.0, 
                             random_state=0)
    
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_X = np.arange(3, 10, 1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])
    plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')
    plt.plot(line_X, line_y_ransac, color='red')   
    plt.xlabel('x-val')
    plt.ylabel('Returns')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'ransac_fit.png', dpi=300)
    plt.close()
    
def polynomial_regression(df, xcols):
    y = df['target_proxy']
    X = df[list(xcols)[0]]
    X = np.transpose(np.array([X]))
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
          
    lr = LinearRegression()
    pr = LinearRegression()
    quadratic = PolynomialFeatures(degree=2)
    X_quad = quadratic.fit_transform(X)
    # fit linear features
    lr.fit(X, y)
    X_fit = np.arange(-2,50,1)[:, np.newaxis]
    y_lin_fit = lr.predict(X_fit)
    
    # fit quadratic features
    pr.fit(X_quad, y)
    y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))
    
    # plot results
    plt.scatter(X, y.values, label='training points')
    plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
    plt.plot(X_fit, y_quad_fit, label='quadratic fit')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'poly_regression.png', dpi=300)
    plt.close()
    
    y_lin_pred = lr.predict(X)
    y_quad_pred = pr.predict(X_quad)
    print('Training MSE linear: %.3f, quadratic: %.3f' % (    
            mean_squared_error(y, y_lin_pred),    
            mean_squared_error(y, y_quad_pred)))    
    print('Training R^2 linear: %.3f, quadratic: %.3f' % (    
            r2_score(y, y_lin_pred),    
            r2_score(y, y_quad_pred)))

def nonlinear(df, xcols):
    y = df['target_proxy']
    X = df[list(xcols)[0]]
    X = np.transpose(np.array([X]))
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
          
    regr = LinearRegression()    
        
    # create quadratic features    
    quadratic = PolynomialFeatures(degree=2)    
    cubic = PolynomialFeatures(degree=3)    
    X_quad = quadratic.fit_transform(X)    
    X_cubic = cubic.fit_transform(X)    
        
    # fit features    
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]    
        
    regr = regr.fit(X, y)    
    y_lin_fit = regr.predict(X_fit)    
    linear_r2 = r2_score(y, regr.predict(X))    
        
    regr = regr.fit(X_quad, y)    
    y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))    
    quadratic_r2 = r2_score(y, regr.predict(X_quad))    
        
    regr = regr.fit(X_cubic, y)    
    y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))    
    cubic_r2 = r2_score(y, regr.predict(X_cubic))    
        
        
    # plot results    
    plt.scatter(X, y, label='training points', color='lightgray')    
        
    plt.plot(X_fit, y_lin_fit,     
             label='linear (d=1), $R^2=%.2f$' % linear_r2,     
             color='blue',     
             lw=2,     
             linestyle=':')    
        
    plt.plot(X_fit, y_quad_fit,     
             label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,    
             color='red',     
             lw=2,    
             linestyle='-')    
        
    plt.plot(X_fit, y_cubic_fit,     
             label='cubic (d=3), $R^2=%.2f$' % cubic_r2,    
             color='green',     
             lw=2,     
             linestyle='--')    
        
    plt.xlabel('x-val')    
    plt.ylabel('Return')    
    plt.legend(loc='best')    
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'nonlinear_regr.png', dpi=300)
    plt.close()
    
    pdb.set_trace()
    # transform features
    X_log = np.log(X)
    y_sqrt = np.sqrt(y)
    
    # fit features
    X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]
    regr = regr.fit(X_log, y_sqrt)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y_sqrt, regr.predict(X_log))
    
    # plot results
    plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')
    plt.plot(X_fit, y_lin_fit, 
             label='linear (d=1), $R^2=%.2f$' % linear_r2, 
             color='blue', 
             lw=2)
    
    plt.xlabel('x-val')
    plt.ylabel('Return')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'sqrt_log.png', dpi=300)

def random_forest_regression(df, xcols):
    y = df['target_proxy']
    X = df[list(xcols)[0]]
    X = np.transpose(np.array([X]))
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X, y)
    sort_idx = X.flatten().argsort()
    lin_regplot(X[sort_idx], y[sort_idx], tree)
    plt.xlabel('x-val')
    plt.ylabel('Return')
    plt.savefig(IMG_PATH + 'tree_regression.png', dpi=300)
    plt.close()
    
    forest = RandomForestRegressor(n_estimators=1000, 
                                   criterion='mse', 
                                   random_state=1, 
                                   n_jobs=-1)
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
    
    plt.scatter(y_train_pred,      
                y_train_pred - y_train,     
                c='black',     
                marker='o',     
                s=35,    
                alpha=0.5,    
                label='Training data')    
    plt.scatter(y_test_pred,      
                y_test_pred - y_test,     
                c='lightgreen',     
                marker='s',     
                s=35,    
                alpha=0.7,    
                label='Test data')    
    plt.xlabel('Predicted values')    
    plt.ylabel('Residuals')    
    plt.legend(loc='best')    
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')    
    plt.xlim([-10, 50])    
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'slr_residuals.png', dpi=300)