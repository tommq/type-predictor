from data import Data
from model import Model

def trainAndValidate(path, solver='lbfgs',cv=4, max_iter=800, winlen=0.01, winstep=0.0025, nfilt=32, nfft=32):
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
        X, y = None, None
    except Exception as e:
        print(e)

# path = '/home/tomas/Documents/School/Master-thesis/grabber/resources/a485-sync/'
path = '/home/tomas/Documents/School/Master-thesis/grabber/resources/a485-sync/compare/'

# for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
# for solver in ['sag']:
#     trainAndValidate(path, solver=solver)

for winlen, winstep, nfilt, nfft in [[0.01, 0.0025, 32, 512], [0.01, 0.0025, 32, 32], [0.01, 0.0025, 24, 32], [0.01, 0.0025, 24, 512], [0.015, 0.0025, 32, 512], [0.015, 0.004, 32, 512], [0.01, 0.002, 32, 512]]:
    trainAndValidate(path, winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft)