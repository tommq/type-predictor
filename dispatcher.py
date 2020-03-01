#   no minimal distance between strokes, simply grabs 100ms of audio with timestamp in the middle
#   no minimal energy check
import numpy as np

from config import Config


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
        before_length = Config.dispatcher_window_size_before / 1000
        after_length = Config.dispatcher_window_size_after / 1000
        # stroke_time = 0.08
        samplerate = 44100

        rem = len(audio_data) % 441
        data = np.array(audio_data[:len(audio_data) - rem]).reshape(-1, 1)  # removing end to make it dividable
        X = []
        Y = []
        sortedKeys = sorted(json.keys())
        for idx, keypress_time in enumerate(sortedKeys):
            if json[keypress_time] in Config.accepted_characters:
                relative_time = keypress_time - start_time
                prev_relative = sortedKeys[idx - 1] - start_time
                gap = (relative_time - before_length) - (prev_relative + after_length)
                if idx > 0 and gap < 0:
                    print("Too little gap between strokes! " + str(gap))
                start_block = int((relative_time - before_length) * samplerate)
                end_block = int((relative_time + after_length) * samplerate)
                extracted_keypress = normalize(data[start_block:end_block])
                X.append(extracted_keypress)
                Y.append(json[keypress_time])
            else:
                print("Character " + str(json[keypress_time]) + " is not accepted")

        return X, Y

    @staticmethod
    def peak_extract(data, labels):
        print("Peak extraction started")
        start_time = None
        if labels:
            start_time = next(key for key, value in labels.items() if value == "start")
            labels.pop(start_time)

        X = []
        # for data in audio_data:
        rem = len(data) % 441
        data = np.array(data[:len(data) - rem])  # removing end to make it dividable
        minimum_interval = 8000  # minimum keypress length
        sample_length = (44100 * 150) / 1000  # size of individual sample = (100ms)
        before_length = (44100 * 10) / 1000
        after_length = (44100 * 140) / 1000
        # dict position:peak sum
        peaks = []
        for x in range(0, len(data) - 440):  # for every datapoint
            peaks.append(np.sum(np.absolute(np.fft.fft(data[x:x + 440]))))  # sum of absolute values returned by fft
            # values of fourier transform result of current 441 data points

        peaks = np.array(peaks)

        tau = np.percentile(peaks, 90)  # 90% of peaks are below this number
        print("found " + str(sum(i > tau for i in peaks)))
        past_x = 0
        step = 10
        x = 0
        events = []
        label_events = []

        sortedKeys = sorted(labels.keys())
        for idx, keypress_time in enumerate(sortedKeys):
            relative_time = (keypress_time - start_time) * 44100
            label_events.append(relative_time)

        while x < peaks.size:
            if peaks[x] >= tau:  # peak must be higher than tau
                if x - past_x >= minimum_interval:  # minimal distance between points/strokes
                    # It is a keypress event (maybe)
                    past_x = x
                    events.append(x)
                    start_block = int((x - before_length))
                    end_block = int((x + after_length))
                    extracted_keypress = normalize(data[start_block:end_block])
                    X.append(extracted_keypress)
                x = past_x + minimum_interval
            else:
                x += step

        for i in range(len(events)):
            if i < len(label_events):
                print(
                    "Guessed : " + str(events[i]) + " real: " + str(
                        int(label_events[i])) + " difference in millis: " + str(
                        int(((label_events[i] - events[i]) / 44100) * 1000)))
            # print("Events", events)
            # print("Label events", label_events)
        return X
