from data import Data
from model import Model
import sys

def trainAndValidate(path, solver='liblinear',cv=4, max_iter=500, winlen=0.01, winstep=0.0025, nfilt=32, nfft=32):
    try:
        data = Data()
        X, y = data.process(path, winlen, winstep, nfilt, nfft)

        # ML Model creating and training
        model_creator = Model(solver, max_iter)
        model_creator.fit_data(X, y)
        score = model_creator.predict_accuracy(X, y, cv)
        accuracy = sum(score)/len(score)
        localVariables = locals()
        print(str(localVariables)[:500], " Avg accuracy: " + str(accuracy))
    except Exception as e:
        print(e)

path = '/home/tomas/PycharmProjects/grabber/resources/a485-sync/699d3629.wav'

# for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
#     trainAndValidate(path, solver=solver)

for winlen, winstep, nfilt, nfft in [[0.01, 0.0025, 32, 512]]:
    trainAndValidate(path, winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft)