#   no minimal distance between strokes, simply grabs 100ms of audio with timestamp in the middle
#   no minimal energy check
import numpy as np


def rms(series):
    return np.math.sqrt(sum(series ** 2) / series.size)


def normalize(series):
    return series / rms(series)


class Dispatcher:

    @staticmethod
    def timestamped_no_peak_check(audio_data, json):

        try:
            start_time = next(key for key, value in json.items() if value == "start")
        except StopIteration:
            print("Start time missing in provided json, should contain timestamp:'start'")
            return None, None
        json.pop(start_time)
        stroke_time = 0.05
        samplerate = 44100
        rem = len(audio_data) % 441
        data = np.array(audio_data[:len(audio_data) - rem]).reshape(-1, 1)  # removing end to make it dividable
        X = []
        Y = []
        for keypress_time in sorted(json.keys()):
            relative_time = keypress_time - start_time
            start_block = int((relative_time.total_seconds() - stroke_time) * samplerate)
            end_block = int((relative_time.total_seconds() + stroke_time) * samplerate)
            extracted_keypress = normalize(data[start_block:end_block])
            X.append(extracted_keypress)
            Y.append(json[keypress_time])

        return X, Y
