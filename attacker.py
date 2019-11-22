from collections import Counter
import csv
from difflib import get_close_matches
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


class Attacker:

    wordlist = list()

    def attack(self, X, y, modelFileName):

        print("In attacker have filename ", modelFileName)
        print(Counter(y))

        model = joblib.load(modelFileName, 'r')
        guesses = model.predict(X)

        self.success_per_character(y, guesses)
        print('guessed', "".join(guesses).split(' '))

        close_words = []
        self.load_words()
        for word in "".join(guesses).split(' '):
            try:
                match = get_close_matches(word, self.wordlist, n=1)[0]
                if len(match) == len(word):
                    close_words.append(match)
                else:
                    close_words.append(word)
            except Exception as e:
                close_words.append(word)


        print('altered', close_words)
        print('real', "".join(y).split(' '))
        print("Accuracy: ", accuracy_score(y, guesses))
        normalized_guesses = list(" ".join(np.asarray(close_words)))
        print("Normalized guesses", normalized_guesses)
        print("Accuracy with dictionary: ", accuracy_score(y, normalized_guesses))
        # print("f1 score: ", f1_score(y, guesses, average='weighted'))
        self.success_per_character(y, normalized_guesses)

    def success_per_character(self, y, guesses):

        success_table = dict()
        for idx, character in enumerate(y):
            if guesses[idx] == character:
                if character in success_table:
                    success_table[character] = success_table[character] + 1
                else:
                    success_table[character] = 1

        counter = Counter(y)

        for character in success_table:
            print(character + " : " + str(success_table[character]) + "/" + str(counter.get(character)))

    def load_words(self):
        with open('resources/words_alpha.txt') as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            for row in csv_reader:
                if len(row) > 0:
                    if "'" not in row[0] and 3 < len(row[0]) < 10:
                        self.wordlist += row
