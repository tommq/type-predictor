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
    classifier = None
    totalWords = 0

    def __init__(self, solver='liblinear', max_iter=150, fecv_folds=5, rfecv_step=50):
        self.classifier = LogisticRegressionCV(solver=solver, class_weight='balanced', max_iter=max_iter, n_jobs=6)
        # self.classifier = MLPClassifier(activation='relu', alpha=1e-5, hidden_layer_sizes=(15,),
        #                                 random_state=1, max_iter=1200)
        # self.classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=12)
        # self.classifier = RandomForestClassifier(n_estimators=50, max_depth=5)
        # self.classifier = RFECV(self.classifier, step=rfecv_step, cv=rfecv_folds, verbose=1, n_jobs=-1)
        print("Selected: ", self.classifier.get_params())
        self.pipeline = make_pipeline(MinMaxScaler(), self.classifier)

    def fit_data(self, X, y):
        print("Going to fit len(X)=", len(X))
        print("Going to fit len(x)=", len(y))
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
