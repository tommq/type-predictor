from collections import Counter
import csv
from difflib import get_close_matches
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from prettytable import PrettyTable
from confusion_matrix_pretty_print import plot_confusion_matrix_from_data
import matplotlib.pyplot as plt


class Attacker:
    wordlist = list()

    def attack(self, X, y=None, modelFileName=None, model=None):

        print("In attacker have filename ", modelFileName)
        if y:
            print(Counter(y))
            if len(y) > len(X):
                y = y[:len(X)]
            elif len(X) > len(y):
                X = X[:len(y)]

        model = joblib.load(modelFileName, 'r') if modelFileName else model
        guesses = model.predict(X)
        probabs, classes = None, None
        try:
            probabs = model.predict_proba(X)
            classes = model.classes_
        except AttributeError:
            print("Probabilities not available for this classifier")

        print('guessed', "".join(guesses).split(' '))

        if y:
            self.success_per_character(y, guesses)

            if probabs and classes:
                top5 = 0.0
                for i, l in enumerate(y[:]):
                    class_prob = probabs[i]
                    top_values = [classes[i] for i in class_prob.argsort()[-5:]]
                    if l in top_values:
                        top5 += 1.0

                print("Top-5 accuracy", top5 / len(y))

        close_words = []
        self.load_words()
        for word in "".join(guesses).split(' '):
            try:
                match = get_close_matches(word, self.wordlist, n=1, cutoff=0.45)[0]
                if len(match) == len(word):
                    close_words.append(match)
                else:
                    close_words.append(word)
            except Exception as e:
                close_words.append(word)

        print('altered', close_words)
        normalized_guesses = list(" ".join(np.asarray(close_words)))
        print("Normalized guesses", normalized_guesses)

        if y:
            print('real', "".join(y).split(' '))
            print("Accuracy: ", accuracy_score(y, guesses))
            print("Recall score: ", recall_score(y, guesses, average='weighted'))
            print("Precision score: ", precision_score(y, guesses, average='weighted'))
            print("F1 score: ", f1_score(y, guesses, average='weighted'))
            print("Accuracy with dictionary: ", accuracy_score(y, normalized_guesses))
            if classes:
                print_confusion_matrix(y, guesses, np.array(classes).tolist())
            self.success_per_character(y, normalized_guesses)

    def success_per_character(self, y, guesses):

        t = PrettyTable(['Predicted character', 'Correctly predicted', 'Prediction success percentage'])
        success_table = dict()
        for idx, character in enumerate(y):
            if guesses[idx] == character:
                if character in success_table:
                    success_table[character] = success_table[character] + 1
                else:
                    success_table[character] = 1

        counter = Counter(y)

        for character in success_table:
            t.add_row([character, str(success_table[character]) + "/" + str(counter.get(character)),
                       str(round(success_table[character] / counter.get(character) * 100, 1)) + '%'])

        print(t.get_string(sortby="Prediction success percentage", reversesort=True))

    def load_words(self):
        with open('resources/words_alpha.txt') as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            for row in csv_reader:
                if len(row) > 0:
                    if "'" not in row[0] and 3 < len(row[0]) < 10:
                        self.wordlist += row


def print_confusion_matrix(y, predictions, classes):

    classes[0] = 'space'
    plot_confusion_matrix_from_data(y, predictions,fz=8, columns=classes)
    plt.show()

