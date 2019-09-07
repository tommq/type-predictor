import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


class Attacker:

    # todo: add dictionary

    def attack(X, y, modelFileName):
        print("In attacker have y ", y)
        print("In attacker have filename ", modelFileName)

        model = joblib.load(modelFileName, 'r')
        guesses = model.predict(X)

        print(guesses)
        print("Accuracy: ", accuracy_score(y, guesses))
        print("f1 score: ", f1_score(y, guesses, average='weighted'))
