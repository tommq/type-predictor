import python_speech_features as sf
import numpy as np


class Extractor:

    @staticmethod
    def transform_mfcc(x):
        print("Going to MFFC size=", len(x))
        arr = np.array([sf.mfcc(sample, 44100, 0.01, 0.0025, 32, 32, preemph=0.9, highfreq=20000, ceplifter=22,
                                appendEnergy=False).flatten() for sample in x])
        return arr
