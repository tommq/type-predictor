#! /usr/bin/python3

from collections import Counter

import numpy as np
import sklearn
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


# todo:  output
# todo: attack mode

#  usage -g/--generate to generate model (use with wav and json)
#       -a/--attack to try to predict words (use with wav only)


class Model:
    pipeline = None
    classifier = None

    def __init__(self, rfecv_folds=5, rfecv_step=50):
        self.classifier = sklearn.linear_model.LogisticRegression(multi_class="auto", solver="liblinear")
        self.classifier = RFECV(self.classifier, step=rfecv_step, cv=rfecv_folds, verbose=1)
        self.pipeline = make_pipeline(MinMaxScaler(), self.classifier)

    def fit_data(self, X, y):
        print("Going to fit len(X)=", len(X))
        print("Going to fit len(x)=", len(y))
        print(Counter(y))
        self.pipeline.fit(X, y)
        print("Fit completed")
        self.predict_accuracy(X, y)

    def predict_accuracy(self, X, y, folds=5):
        print("Predicting accuracy")
        print("score: ", np.mean(cross_val_score(self.pipeline, X, y, cv=folds)))

    def predict(self, X):
        print(self.pipeline.predict(X))
