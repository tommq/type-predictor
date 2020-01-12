from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.svm import LinearSVC

from data import Data
from model import Model

path = '/home/tomas/Documents/School/Master-thesis/grabber/resources/a485-sync/closet/selected/'




def trainAndValidate(path, classifier=None, solver='lbfgs', cv=4, max_iter=800, winlen=0.04, winstep=0.005, numcep=32,
                     nfilt=96, nfft=2048, X=None, y=None):
    try:
        # data = Data()
        # X, y = data.process(path, winlen, winstep, numcep, nfilt, nfft)

        # ML Model creating and training
        model_creator = Model(classifier=classifier, solver=solver, max_iter=max_iter)
        model_creator.fit_data(X, y)
        score = model_creator.predict_accuracy(X, y, cv)
        accuracy = sum(score) / len(score)
        localVariables = locals()
        print(str(localVariables)[:500], " Avg accuracy: " + str(accuracy))
        X, y = None, None
    except Exception as e:
        print(e)


# path = '/home/tomas/Documents/School/Master-thesis/grabber/resources/a485-sync/'

# for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
# for solver in ['sag']:
#     trainAndValidate(path, solver=solver)
#
# for winlen, winstep, numcep, nfilt, nfft in [[0.02, 0.0025, 32, 96, 2048], [0.03, 0.0025, 32, 96, 2048],
#                                      [0.02, 0.0025, 16, 96, 2048], [0.02, 0.0025, 32, 128, 2048],
#                                      [0.04, 0.005, 32, 96, 4096], [0.01, 0.0025, 32, 96, 2048],
#                                      [0.02, 0.005, 32, 96, 2048]]:
#     trainAndValidate(path, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft)

classifiers = []
# classifiers.append(MLPClassifier(activation='relu', alpha=1e-5, hidden_layer_sizes=(2048,),random_state=1, max_iter=2000)) #Avg accuracy: 0.7945881045466272
# classifiers.append(BernoulliRBM(n_components=1024, n_iter=2000))
# classifiers.append(SGDClassifier(class_weight='balanced', loss="hinge", penalty="l2", max_iter=2000))
classifiers.append(RandomForestClassifier(class_weight='balanced', n_estimators=5000, max_depth=15, n_jobs=6))
# classifiers.append(ExtraTreesClassifier(class_weight='balanced', n_estimators=5000, max_depth=5,  random_state=0))
# classifiers.append(LinearSVC(class_weight='balanced', tol=1e-5))
# classifiers.append(LogisticRegressionCV(solver='lbfgs', class_weight='balanced', max_iter=1200, n_jobs=8))
data = Data()
X, y = data.process(path, winlen=0.04, winstep=0.005, numcep=32, nfilt=96, nfft=2048)

for classifier in classifiers:
    trainAndValidate(path=path, classifier=classifier, X=X, y=y)
