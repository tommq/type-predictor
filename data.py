import glob
import os

from dispatcher import Dispatcher
from extractor import Extractor
from importer import Importer


class Data:

    total_X = []
    total_y = []

    def process(self, path):
        if os.path.isdir(path):
            self.process_folder(path)
        elif os.path.isfile(path):
            self.process_file(path)
        else:
            print("Incorrect path")
            exit(1)
        return self.total_X, self.total_y

    def process_folder(self, path):
        files = glob.glob(path + "*.wav")
        for file_path in files:
            self.process_file(file_path)
            print("Adding" + file_path)

    def process_file(self, wav_file_path):
        wav_file, json_file = Importer.load(wav_file_path)
        if not wav_file or not json_file:
            print("Error for path " + wav_file_path)
            return

        X, y = Dispatcher.timestamped_no_peak_check(wav_file, json_file)
        if not X or not y:
            print("X not present, exiting...")
            return
        X = Extractor.transform_mfcc(X)
        self.total_X.extend(X)
        self.total_y.extend(y)