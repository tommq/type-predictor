import python_speech_features as sf
import numpy as np


class Extractor:

    @staticmethod
    def transform_mfcc(x, winlen=0.01, winstep=0.0025, numcep=13, nfilt=32, nfft=512):
        print("Going to MFFC size=", len(x))
        arr = np.array([sf.mfcc(sample, 44100, winlen, numcep, winstep, nfilt, nfft, preemph=0.9, highfreq=20000, ceplifter=22,
                                appendEnergy=False).flatten() for sample in x])
        print("Features: " + str(arr.shape))
        return arr
