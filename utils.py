from datetime import datetime as dt
import random
from config import Config


def get_timestamp():
    now = dt.now()
    return now


def dt_from_str(string_input):
    return dt.strptime(string_input, '%Y-%m-%d %H:%M:%S.%f')


def getN(X, y, n):
    newX = []
    newY = []

    bag = {}

    for idx, sample in enumerate(y):
        bag[idx] = sample

    stop = n * len(Config.accepted_characters)
    chosen = {}
    counter = 0

    indexes = list(bag.keys())

    while counter < stop:
        index = random.choice(indexes)

        letter = bag[index]

        if letter not in chosen:
            chosen[letter] = 0

        if chosen[letter] < n:
            newX.append(X[index])
            newY.append(bag[index])
            chosen[letter] = chosen[letter] + 1
            bag.pop(index)
            indexes.remove(index)
            counter = counter + 1

    return newX, newY
