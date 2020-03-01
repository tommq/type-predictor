import glob
import os

from dispatcher import Dispatcher
from feature_extractor import Extractor
from importer import Importer


class Data:
    total_X = []
    total_y = []
    winlen, winstep, nfilt, nfft, numcep = None, None, None, None, None

    def __init__(self):
        self.total_X = []
        self.total_y = []
        self.winlen, self.winstep, self.nfilt, self.nfft, self.numcep = None, None, None, None, None

    def process(self, path, winlen=0.08, winstep=0.005, numcep=32, nfilt=96, nfft=4096, attack=False, peak=False):
        self.winlen = winlen
        self.winstep = winstep
        self.numcep = numcep
        self.nfilt = nfilt
        self.nfft = nfft
        if os.path.isdir(path):
            self.process_folder(path, attack=attack, peak=peak)
        elif os.path.isfile(path):
            self.process_file(path, attack=attack, peak=peak)
        else:
            print("Incorrect path" + path)
            exit(1)
        return self.total_X, self.total_y

    def process_folder(self, path, attack=False, peak=False):
        files = glob.glob(path + "*.wav")
        for file_path in files:
            self.process_file(file_path, attack=attack, peak=peak)
            print("Adding" + file_path)

    def process_file(self, wav_file_path, attack=False, peak=False):
        wav_file, json_file = Importer.load(wav_file_path)
        if not wav_file:
            print("Error for path " + wav_file_path)
            return
        if not json_file and not attack:
            print("Error for path " + wav_file_path)
            return

        # if attack and not json_file:
        if attack and peak:
            X = Dispatcher.peak_extract(wav_file, json_file)
            if json_file:
                y = [json_file[i] for i in sorted(json_file.keys())]
                y.pop(0)
            else:
                y = None
        else:
            X, y = Dispatcher.timestamped_no_peak_check(wav_file, json_file)

        if X is None:
            print("X not present, exiting...")
            return
        X = Extractor.transform_mfcc(X, self.winlen, self.winstep, self.numcep, self.nfilt, self.nfft)
        self.total_X.extend(X)

        if y:
            self.total_y.extend(y)
