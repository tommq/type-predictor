#! /usr/bin/python3

from collections import Counter

import numpy as np
import sklearn
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
import joblib


# todo:  output
# todo: experiment with different feature extractions, timeframes for audio, etc. Two fold cross val or so


#  usage -g/--generate to generate model (use with wav and json)
#       -a/--attack to try to predict words (use with wav only)


class Model:
    pipeline = None
    classifier = None
    totalWords = 0

    def __init__(self, rfecv_folds=5, rfecv_step=50):
        self.classifier = sklearn.linear_model.LogisticRegression(multi_class="auto", solver="liblinear")
        # self.classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=12)
        self.classifier = RFECV(self.classifier, step=rfecv_step, cv=rfecv_folds, verbose=1, n_jobs=1)
        self.pipeline = make_pipeline(MinMaxScaler(), self.classifier)

    def fit_data(self, X, y):
        print("Going to fit len(X)=", len(X))
        print("Going to fit len(x)=", len(y))
        print(Counter(y))
        self.pipeline.fit(X, y)
        self.totalWords = y.__len__()
        print("Fit completed")

    def predict_accuracy(self, X, y, folds=5):
        print("Predicting accuracy")
        print("score: ", np.mean(cross_val_score(self.pipeline, X, y, cv=folds, n_jobs=-2)))

    def dump(self, name):
        joblib.dump(self.pipeline, name + str(self.totalWords) + "W" + ".model")

    def predict_probability(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)
