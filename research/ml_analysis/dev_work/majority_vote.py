"""
Module to describe an ensemble classifier
"""
from itertools import product
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import _name_estimators, Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.externals import six
from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.ml_utils import IMG_PATH
mpl.use('Agg')


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """
    A majority vote ensemble classifier

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
        """
        Constructor
        """
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
        self.lablenc_ = LabelEncoder()
        self.classifiers_ = []
        self.classes_ = []

    def fit(self, x_vals, y_vals):
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
            raise ValueError("vote must be 'probability' or 'classlabel'; got (vote=%r)"
                             % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))

        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_.fit(y_vals)
        self.classes_ = self.lablenc_.classes_
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(x_vals, self.lablenc_.transform(y_vals))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, x_vals):
        """
        Predict class labels for X.

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
            maj_vote = np.argmax(self.predict_proba(x_vals), axis=1)
        else:
            # 'classlabel' vote
            #  Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(x_vals) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x:
                                           np.argmax(np.bincount(x, weights=self.weights)),
                                           axis=1, arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, x_vals):
        """
        Predict class probabilities for X.

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
        probas = np.asarray([clf.predict_proba(x_vals) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """
        Get classifier parameter names for GridSearch
        """
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        out = self.named_classifiers.copy()
        for name, step in six.iteritems(self.named_classifiers):
            for key, value in six.iteritems(step.get_params(deep=True)):
                out['%s__%s' % (name, key)] = value
        return out


def majority_vote():
    """
    Example for testing majority vote class
    """
    iris = datasets.load_iris()
    x_vals, y_vals = iris.data[50:, [1, 2]], iris.target[50:]
    labenc = LabelEncoder()
    y_vals = labenc.fit_transform(y_vals)
    x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals,
                                                        test_size=0.5, random_state=1)

    clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
    clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
    clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])
    clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']

    # Majority Rule (hard) Voting
    mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

    clf_labels += ['Majority Voting']
    all_clf = [pipe1, clf2, pipe3, mv_clf]
    print('10-fold cross validation:\n')
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf, X=x_train, y=y_train, cv=10, scoring='roc_auc')
        print("ROC AUC: %0.2f (+/- %0.2f) [%s]"  % (scores.mean(), scores.std(), label))

    colors = ['black', 'orange', 'blue', 'green']
    linestyles = [':', '--', '-.', '-']
    for clf, label, clr, lin_style in zip(all_clf, clf_labels, colors, linestyles):
        # assuming the label of the positive class is 1
        y_pred = clf.fit(x_train, y_train).predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_pred)
        print(y_pred)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(fpr, tpr, color=clr, linestyle=lin_style,
                 label='%s (auc = %0.2f)' % (label, roc_auc))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'roc.png', dpi=300)
    plt.close()

    stdc = StandardScaler()
    x_train_std = stdc.fit_transform(x_train)
    all_clf = [pipe1, clf2, pipe3, mv_clf]
    x_min = x_train_std[:, 0].min() - 1
    x_max = x_train_std[:, 0].max() + 1
    y_min = x_train_std[:, 1].min() - 1
    y_max = x_train_std[:, 1].max() + 1
    xxx, yyy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    _, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(7, 5))
    for idx, clf, ttt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
        clf.fit(x_train_std, y_train)
        z_vals = clf.predict(np.c_[xxx.ravel(), yyy.ravel()])
        z_vals = z_vals.reshape(xxx.shape)
        axarr[idx[0], idx[1]].contourf(xxx, yyy, z_vals, alpha=0.3)
        axarr[idx[0], idx[1]].scatter(x_train_std[y_train == 0, 0], x_train_std[y_train == 0, 1],
                                      c='blue', marker='^', s=50)
        axarr[idx[0], idx[1]].scatter(x_train_std[y_train == 1, 0], x_train_std[y_train == 1, 1],
                                      c='red', marker='o', s=50)
        axarr[idx[0], idx[1]].set_title(ttt)
    plt.text(-3.5, -4.5, s='Sepal width [standardized]', ha='center', va='center', fontsize=12)
    plt.text(-10.5, 4.5, s='Petal length [standardized]', ha='center', va='center',
             fontsize=12, rotation=90)
    plt.tight_layout()
    plt.savefig(IMG_PATH + 'voting_panel.png', bbox_inches='tight', dpi=300)
    # print(mv_clf.get_params())
    params = {'decisiontreeclassifier__max_depth': [1, 2],
              'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
    grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring='roc_auc')
    grid.fit(x_train, y_train)

    for params, mean_score, scores in grid.cv_results_:
        print("%0.3f+/-%0.2f %r" % (mean_score, scores.std() / 2, params))
    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.2f' % grid.best_score_)

if __name__ == '__main__':
    majority_vote()
