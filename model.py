from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import joblib


class Model:
    pipeline = None
    totalWords = 0

    def __init__(self, classifier=None, max_iter=1500):
        if not classifier:
            classifier = MLPClassifier(activation='relu', alpha=1e-5, hidden_layer_sizes=(2048,), random_state=1,
                                       max_iter=max_iter)

        print("Selected classifier: ", classifier.get_params())
        self.pipeline = make_pipeline(MinMaxScaler(), classifier)

    def fit_data(self, X, y):
        print("Going to fit len(X)=", len(X), "len(y)=", len(y))
        print(Counter(y))
        self.pipeline.fit(X, y)
        print("Fit completed")
        self.totalWords = len(y)

    def predict_accuracy(self, X, y, folds=5):
        print("Predicting accuracy")
        score = cross_val_score(self.pipeline, X, y, cv=folds, n_jobs=8)
        print("score: ", score)
        return score

    def dump(self, name):
        model = self.pipeline.steps[1][0]
        joblib.dump(self.pipeline, name + str(self.totalWords) + "w-" + model + ".model")

    def predict_probability(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)
