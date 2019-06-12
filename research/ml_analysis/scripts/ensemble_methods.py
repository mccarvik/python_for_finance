import sys, datetime, pdb, time, operator
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product

from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.externals import six
from sklearn.pipeline import _name_estimators, Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from utils.ml_utils import plot_decision_regions, standardize, IMG_PATH


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """ A majority vote ensemble classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble

    vote : str, {'classlabel', 'probability'} (default='label')
      If 'classlabel' the prediction is based on the argmax of
        class labels. Else if 'probability', the argmax of
        the sum of probabilities is used to predict the class label
        (recommended for calibrated classifiers).

    weights : array-like, shape = [n_classifiers], optional (default=None)
      If a list of `int` or `float` values are provided, the classifiers
      are weighted by importance; Uses uniform weights if `weights=None`.

    """
    
    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers.

            Parameters
            ----------
            X : {array-like, sparse matrix}, shape = [n_samples, n_features]
                Matrix of training samples.
    
            y : array-like, shape = [n_samples]
                Vector of target class labels.
    
            Returns
            -------
            self : object

        """
        
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'; got (vote=%r)" % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))

        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        Returns
        ----------
        maj_vote : array-like, shape = [n_samples]
            Predicted class labels.
            
        """
        
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' vote

            #  Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg_proba : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.

        """
        
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out


def majority_vote(df, xcols):
    y = df['target']
    X = df[list(xcols)]
    
    # Need this just for specific cases, need postive results to be a value of 1
    y = y.map({4:1, 0:0})
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)

    clf1 = LogisticRegression(penalty='l2', 
                              C=0.001, 
                              random_state=0)
    clf2 = DecisionTreeClassifier(max_depth=1, 
                                  criterion='entropy', 
                                  random_state=0)
    clf3 = KNeighborsClassifier(n_neighbors=1, 
                                p=2, 
                                metric='minkowski')
    pipe1 = Pipeline([['sc', StandardScaler()],
                      ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()],
                      ['clf', clf3]])
    clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
    
    # Majority Rule (hard) Voting
    mv_clf = MajorityVoteClassifier(
                    classifiers=[pipe1, clf2, pipe3])
    
    clf_labels += ['Majority Voting']
    all_clf = [pipe1, clf2, pipe3, mv_clf]
    print('10-fold cross validation:\n')
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf, 
                                 X=X_train, 
                                 y=y_train, 
                                 cv=10, 
                                 scoring='roc_auc')
        print("ROC AUC: %0.2f (+/- %0.2f) [%s]"  % (scores.mean(), scores.std(), label))
    
    colors = ['black', 'orange', 'blue', 'green']
    linestyles = [':', '--', '-.', '-']
    for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
        # assuming the label of the positive class is 1
        y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test, 
                                         y_score=y_pred)
        print(y_pred)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(fpr, tpr, 
                 color=clr, 
                 linestyle=ls, 
                 label='%s (auc = %0.2f)' % (label, roc_auc))
    
    plt.legend(loc='best')
    plt.plot([0, 1], [0, 1], 
             linestyle='--', 
             color='gray', 
             linewidth=2)
    
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'roc.png', dpi=300)
    plt.close()
    
    # sc = StandardScaler()
    # X_train_std = sc.fit_transform(X_train)
        
    all_clf = [pipe1, clf2, pipe3, mv_clf]    
    x_min = X_train[:, 0].min() - 1    
    x_max = X_train[:, 0].max() + 1    
    y_min = X_train[:, 1].min() - 1    
    y_max = X_train[:, 1].max() + 1    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),    
                         np.arange(y_min, y_max, 0.1))    
    f, axarr = plt.subplots(nrows=2, ncols=2,     
                            sharex='col',     
                            sharey='row',     
                            figsize=(7, 5))    
    for idx, clf, tt in zip(product([0, 1], [0, 1]),    
                            all_clf, clf_labels):    
        clf.fit(X_train, y_train)    
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])    
        Z = Z.reshape(xx.shape)
        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)    
        axarr[idx[0], idx[1]].scatter(X_train[y_train.values==0, 0],     
                                      X_train[y_train.values==0, 1],     
                                      c='blue',     
                                      marker='^',    
                                      s=50)    
        axarr[idx[0], idx[1]].scatter(X_train[y_train.values==1, 0],     
                                      X_train[y_train.values==1, 1],     
                                      c='red',     
                                      marker='o',    
                                      s=50)    
        axarr[idx[0], idx[1]].set_title(tt)    
    plt.text(-3.5, -4.5,     
             s=xcols[0],     
             ha='center', va='center', fontsize=12)    
    plt.text(-10.5, 4.5,     
             s=xcols[1],     
             ha='center', va='center',     
             fontsize=12, rotation=90)    
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'voting_panel.png', bbox_inches='tight', dpi=300)
    # print(mv_clf.get_params())
    
    params = {'decisiontreeclassifier__max_depth': [1, 2],    
              'pipeline-1__clf__C': [0.001, 0.1, 100.0]}    
        
    grid = GridSearchCV(estimator=mv_clf,     
                        param_grid=params,     
                        cv=10,     
                        scoring='roc_auc')    
    grid.fit(X_train, y_train)    
        
    for params, mean_score, scores in grid.grid_scores_:    
        print("%0.3f+/-%0.2f %r" % (mean_score, scores.std() / 2, params))
    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.2f' % grid.best_score_)


def bagging(df, xcols):
    y = df['target']
    X = df[list(xcols)]
    
    # Need this just for specific cases, need postive results to be a value of 1
    y = y.map({4:1, 0:0})
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
    
    tree = DecisionTreeClassifier(criterion='entropy',     
                                  max_depth=None,    
                                  random_state=1)    
    bag = BaggingClassifier(base_estimator=tree,    
                            n_estimators=500,     
                            max_samples=1.0,     
                            max_features=1.0,     
                            bootstrap=True,     
                            bootstrap_features=False,     
                            n_jobs=1,     
                            random_state=1)
    
       
        
    tree = tree.fit(X_train, y_train)    
    y_train_pred = tree.predict(X_train)    
    y_test_pred = tree.predict(X_test)    
    
    tree_train = accuracy_score(y_train, y_train_pred)    
    tree_test = accuracy_score(y_test, y_test_pred)    
    print('Decision tree train/test accuracies %.3f/%.3f'    
          % (tree_train, tree_test))    
        
    bag = bag.fit(X_train, y_train)    
    y_train_pred = bag.predict(X_train)    
    y_test_pred = bag.predict(X_test)    
        
    bag_train = accuracy_score(y_train, y_train_pred)     
    bag_test = accuracy_score(y_test, y_test_pred)     
    print('Bagging train/test accuracies %.3f/%.3f'    
          % (bag_train, bag_test))
    
    x_min = X_train[:, 0].min() - 1    
    x_max = X_train[:, 0].max() + 1    
    y_min = X_train[:, 1].min() - 1    
    y_max = X_train[:, 1].max() + 1    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),    
                         np.arange(y_min, y_max, 0.1))    
    f, axarr = plt.subplots(nrows=1, ncols=2,     
                            sharex='col',     
                            sharey='row',     
                            figsize=(8, 3))    
        
    for idx, clf, tt in zip([0, 1],    
                            [tree, bag],    
                            ['Decision Tree', 'Bagging']):    
        clf.fit(X_train, y_train)    
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])    
        Z = Z.reshape(xx.shape)    
        axarr[idx].contourf(xx, yy, Z, alpha=0.3)    
        axarr[idx].scatter(X_train[y_train.values==0, 0],     
                           X_train[y_train.values==0, 1],     
                           c='blue', marker='^')    
        axarr[idx].scatter(X_train[y_train.values==1, 0],     
                           X_train[y_train.values==1, 1],     
                           c='red', marker='o')    
        axarr[idx].set_title(tt)    
        
    pdb.set_trace()
    axarr[0].set_ylabel(xcols[1], fontsize=12)    
    plt.text(10.2, -1.2,     
             s=xcols[0],     
             ha='center', va='center', fontsize=12)    
    plt.tight_layout()    
    plt.savefig(IMG_PATH + 'voting_panel_bagging.png', bbox_inches='tight', dpi=300)   


def adaboost(df, xcols):
    y = df['target']
    X = df[list(xcols)]
    
    # Need this just for specific cases, need postive results to be a value of 1
    y = y.map({4:1, 0:0})
    
    # Standardize and split the training nad test data
    X_std = standardize(X)
    ts = 0.3
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=0)
          
    tree = DecisionTreeClassifier(criterion='entropy', 
                                  max_depth=1,
                                  random_state=0)
    
    ada = AdaBoostClassifier(base_estimator=tree,
                             n_estimators=500, 
                             learning_rate=0.1,
                             random_state=0)
    
    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    
    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print('Decision tree train/test accuracies %.3f/%.3f'
          % (tree_train, tree_test))
    
    ada = ada.fit(X_train, y_train)
    y_train_pred = ada.predict(X_train)
    y_test_pred = ada.predict(X_test)
    
    ada_train = accuracy_score(y_train, y_train_pred) 
    ada_test = accuracy_score(y_test, y_test_pred) 
    print('AdaBoost train/test accuracies %.3f/%.3f'
          % (ada_train, ada_test))
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(8, 3))
    
    for idx, clf, tt in zip([0, 1],
                            [tree, ada],
                            ['Decision Tree', 'AdaBoost']):
        clf.fit(X_train, y_train)
    
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        axarr[idx].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx].scatter(X_train[y_train.values == 0, 0],
                           X_train[y_train.values == 0, 1],
                           c='blue', marker='^')
        axarr[idx].scatter(X_train[y_train.values == 1, 0],
                           X_train[y_train.values == 1, 1],
                           c='red', marker='o')
        axarr[idx].set_title(tt)
    
    axarr[0].set_ylabel(xcols[0], fontsize=12)
    plt.text(10.2, -1.2,
             s=xcols[1],
             ha='center', va='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'adaboost.png', bbox_inches='tight', dpi=300)   
