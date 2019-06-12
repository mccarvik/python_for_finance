import pdb
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from itertools import combinations

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
      self.scoring = scoring
      # self.estimator = clone(estimator)
      self.estimator = estimator
      self.k_features = k_features
      self.test_size = test_size
      self.random_state = random_state
      self.removed_order = []
    
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
    
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]
    
        while dim > self.k_features:
            scores = []
            subsets = []
    
            # find all combinations with one item removed
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
    
            # find the subset that did best, set the subset to it
            # ipso facto the worst feature is removed, in theory
            # also keep track of removal order
            best = np.argmax(scores)
            self.removed_order.append(tuple(set(subsets[best]) ^ set(self.indices_))[0])
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
    
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self
    
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score