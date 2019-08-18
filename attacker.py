import joblib


class Attacker:

    def attack(X, y, modelFileName):

        print("In attacker have X ", X)
        print("In attacker have y ",y)
        print("In attacker have filename ", modelFileName)

        model = joblib.load(modelFileName, 'rb')
        print("Got model:", model)



        # parse target
        # show actual words, make sure at least spaces are correct

        # predict
        # predisc whole words

        # show results

