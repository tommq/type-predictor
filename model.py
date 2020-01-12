#! /usr/bin/python3

from collections import Counter

import numpy as np
import sklearn
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
import joblib


# todo:  output
# todo: experiment with different feature extractions, timeframes for audio, etc. Two fold cross val or so


#  usage -g/--generate to generate model (use with wav and json)
#       -a/--attack to try to predict words (use with wav only)


class Model:
    pipeline = None
    # classifier = None
    totalWords = 0

    def __init__(self, classifier=None, solver='lbfgs', max_iter=1500):

        # self.pipeline = classifier
        if not classifier:
            classifier = LogisticRegressionCV(solver=solver, class_weight='balanced', max_iter=max_iter, n_jobs=6)
        #
        print("Selected: ", classifier.get_params())
        self.pipeline = make_pipeline(MinMaxScaler(), classifier)

    def fit_data(self, X, y):
        print("Going to fit len(X)=", len(X), "len(y)=", len(y))
        print(Counter(y))
        self.pipeline.fit(X, y)
        self.totalWords = y.__len__()
        print("Fit completed")

    def predict_accuracy(self, X, y, cv=4):
        print("Predicting accuracy")
        score = cross_val_score(self.pipeline, X, y, cv=cv, n_jobs=4)
        print("score: ", score)
        return score

    def dump(self, name):
        model = self.pipeline.steps[1][0]
        joblib.dump(self.pipeline, name + str(self.totalWords) + "w-" + model + ".model")

    def predict_probability(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)
