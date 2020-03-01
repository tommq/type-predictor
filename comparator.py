import sys
import traceback

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from attacker import Attacker
from data import Data
from model import Model

train_path = '/home/tomas/Documents/School/Master-thesis/grabber/resources/a485-sync/fifth/'
test_path = '/home/tomas/Documents/School/Master-thesis/grabber/resources/a485-sync/closet/selected/'
attack_audio_path = '/home/tomas/Documents/School/Master-thesis/grabber/resources/a485-sync/closet/selected/18abea94.wav'


def trainAndCrossValidate(classifier=None, train_X=None, train_y=None, folds=5):
    try:
        print("Predicting accuracy")
        score = cross_val_score(classifier, X=train_X, y=train_y, cv=folds, n_jobs=8)
        print(str(folds) + " fold cross validation score: " + str(score))

    except Exception as e:
        print("Exception in train and CV: ", e)


def trainAndAttack(classifier=None, train_X=None, train_y=None, test_X=None, test_y=None):
    try:
        model_creator = Model(classifier=classifier)
        model_creator.fit_data(train_X, train_y)
        predictions = model_creator.predict(test_X)
        accuracy = accuracy_score(test_y, predictions)
        print("Accuracy for " + str(classifier) + " : " + str(accuracy))
    except Exception as e:
        print("Exception in train and attack: ", e, )


def trainAndAttackRaw(classifier=None, train_X=None, train_y=None, audioPath=None):
    try:
        print("Imported data " + str(len(train_X)) + " and " + str(len(train_y)))
        testData = Data()
        test_X, test_y = testData.process(audioPath, attack=True, peak=True)
        model_creator = Model(classifier=classifier)
        print("Imported data " + str(len(train_X)) + " and " + str(len(train_y)))
        model_creator.fit_data(train_X, train_y)
        attacker = Attacker()
        attacker.attack(X=test_X, y=test_y, model=model_creator)
    except Exception as e:
        print("Exception in train and attack raw: ", e.__cause__)
        traceback.print_exc(file=sys.stdout)


# for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
# for solver in ['sag']:
#     trainAndValidate(path, solver=solver)
#
# for winlen, winstep, numcep, nfilt, nfft in [[0.02, 0.0025, 32, 96, 2048], [0.03, 0.0025, 32, 96, 2048],
#                                      [0.02, 0.0025, 16, 96, 2048], [0.02, 0.0025, 32, 128, 2048],
#                                      [0.04, 0.005, 32, 96, 4096], [0.01, 0.0025, 32, 96, 2048],
#                                      [0.02, 0.005, 32, 96, 2048]]:
#     trainAndValidate(path, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft)


mlp = MLPClassifier(activation='relu', alpha=1e-5, hidden_layer_sizes=(2048,), random_state=1, max_iter=1500)
lr = LogisticRegressionCV(solver='lbfgs', class_weight='balanced', max_iter=1500, n_jobs=8)
svc = LinearSVC(class_weight='balanced', tol=1e-5, max_iter=10000)

data = Data()
X, y = data.process(train_path, winlen=0.08, winstep=0.005, numcep=32, nfilt=96, nfft=4096)

print("Imported data " + str(len(X)) + " and " + str(len(y)))

# testdata = Data()

# testX, testy = testdata.process(test_path, winlen=0.08, winstep=0.005, numcep=32, nfilt=96, nfft=4096)

# trainAndCrossValidate(classifier=svc, train_X=X, train_y=y)
trainAndAttackRaw(classifier=mlp, train_X=X, train_y=y, audioPath=attack_audio_path)
# trainAndAttack(classifier=lr, train_X=X, train_y=y, test_X=testX, test_y=testy)
# trainAndAttack(classifier=mlp, train_X=X, train_y=y, test_X=testX, test_y=testy)
# for size in [10, 20, 30, 50, 70, 90]:
#     newX, newY = getN(X, y, size)
#     print(Counter(newY))
#
#     for classifier in classifiers:
#         trainAndValidate(path=path, classifier=classifier, X=newX, y=newY)
