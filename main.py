#!/usr/bin/env python3

import argparse
from attacker import Attacker
from data import Data
from model import Model

if __name__ == "__main__":

    # Argument parsing

    parser = argparse.ArgumentParser()

    parser.add_argument("--wav", "-wav", type=str, required=True, help='Wav file name')
    parser.add_argument("--attack", "-a", type=str, required=False,
                        help='Attack audio with pre-trained model')
    parser.add_argument('--folds', type=int, default=5, help='How many folds to use for cross-validation')
    parser.add_argument('--nocv', type=int, default=False, help='Disables cross-validation')

    args = parser.parse_args()
    if not args.wav:
        print("Can't work without wav file!")

    # FILE IMPORT

    data = Data()
    X, y = data.process(args.wav, attack=bool(args.attack))

    # attack mode
    if args.attack:
        print(args.attack)
        attacker = Attacker()
        attacker.attack(X, y, args.attack)

    # train mode
    else:
        # ML Model creating and training
        model_creator = Model()
        model = model_creator.fit_data(X, y)
        model_creator.dump(args.wav)
        print("Dumped")
        if not args.nocv:
            model_creator.predict_accuracy(X, y)
