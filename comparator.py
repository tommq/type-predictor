from data import Data
from model import Model

def trainAndValidate(path):

    data = Data()
    X, y = data.process(path)

    # ML Model creating and training
    model_creator = Model()
    model_creator.fit_data(X, y)
    model_creator.predict_accuracy(X, y)


