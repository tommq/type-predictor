import numpy as np
from dst.libraries import al
import warnings
from datetime import datetime as dt

warnings.filterwarnings("ignore")


def offline(in_queue, out_queue, display_queue, config):
    """

    :param in_queue: Queue to receive audio file
    :type in_queue: multiprocessing.Queue
    :param out_queue: Queue where to put extracted keypress samples
    :type out_queue: multiprocessing.Queue
    :param display_queue: Queue where to put visual information to be displayed
    :type display_queue: multiprocessing.Queue
    :param config: a Config object
    :type config: Config
    :return: None
    :rtype:
    """
    for data in iter(in_queue.get, None):
        rem = len(data) % 441
        data = np.array(data[:len(data) - rem]) # removing end to make it dividable
        minimum_interval = config.dispatcher_min_interval # minimum keypress length
        sample_length = (44100 * config.dispatcher_window_size) / 1000 # size of individual sample = (100ms)

        persistence = config.dispatcher_persistence

        peaks = []
        for x in range(0, len(data) - 440): # for every datapoint
            peaks.append(np.sum(np.absolute(np.fft.fft(data[x:x + 440]))))  # sum of absolute values returned by fft
            # values of fourier transform result of current 441

        peaks = np.array(peaks)
        tau = np.percentile(peaks, config.dispatcher_threshold) # 90% of peaks are below this number

        x = 0   # iterator
        events = []
        step = config.dispatcher_step_size
        past_x = - minimum_interval - step
        idx = 0
        while x < peaks.size:
            if peaks[x] >= tau: # peak must me higher than tau
                if x - past_x >= minimum_interval:  # minimal distance between points/strokes
                    # It is a keypress event (maybe)
                    keypress = al.normalize(data[x:x + sample_length])  # extract next 100ms
                    past_x = x
                    # Pass it immediately to workers
                    out_queue.put([idx, keypress])
                    idx += 1
                    # Display event point
                    # display_queue.put(x)
                    events.append(keypress)
                x = past_x + minimum_interval
            else:
                x += step

        display_queue.put(len(events))
        # If persistence, save stuff to path
        # TODO implement it
        if persistence:
            pass

    # Send termination flag to workers
    for _x in xrange(config.workers):
        out_queue.put((-1, None))


# method goes through timestamps and checks for peaks in those periods
# found keypresses are sent to miner, missed timestamps shall be deleted from txt file

def fromTimestamp(timestamp):
    pass

def dt_from_str(string_input):
    return dt.strptime(string_input, '%Y-%m-%d %H:%M:%S.%f')


#   no minimal distance between strokes, simply grabs 100ms of audio with timestamp in the middle
#   no minimal energy check
def timestamped_no_peak_check(in_queue, out_queue, display_queue, txtq, json, config):

    print("Timestamp no peak check started")
    start_time = json.keys()[json.values().index("start")]
    json.pop(start_time)
    stroke_time = 0.05
    samplerate = 44100
    data = in_queue.get()
    rem = len(data) % 441
    data = np.array(data[:len(data) - rem])  # removing end to make it dividable
    minimum_interval = config.dispatcher_min_interval  # minimum keypress length
    sample_length = (samplerate * config.dispatcher_window_size) / 1000  # size of individual sample = (100ms)
    x = 0  # iterator
    events = []
    step = config.dispatcher_step_size
    past_x = - minimum_interval - step
    idx = 0

    for keypress in sorted(json.keys()):
        relative_time = keypress - start_time
        start_block = int((relative_time.total_seconds() - stroke_time) * samplerate)
        end_block = int((relative_time.total_seconds() + stroke_time) * samplerate)
        keypress = al.normalize(data[start_block:end_block])
        out_queue.put([idx, keypress])
        idx += 1
        events.append(keypress)

    display_queue.put(len(events))

    #   generate and save txt file
    spaced = ""
    for timestamp in sorted(json):
        character = json[timestamp]
        spaced = spaced + character + "\n"
        spaced = spaced.replace(" ", "*")
    # print(spaced)
    txtq.put(spaced)
    txtq.put(None)

    # Send termination flag to workers
    for _x in xrange(config.workers):
        out_queue.put((-1, None))
    print("Timestamp no peak check finished")



def generate_txt_file():
    pass

def timestamped(in_queue, out_queue, display_queue, config, json):
    """
dispatcher with a json file containing true values and timestamps
    :param in_queue:
    :param out_queue:
    :param display_queue:
    :param config:
    :param json:
    """
    for data in iter(in_queue.get, None):
        rem = len(data) % 441
        data = np.array(data[:len(data) - rem])  # removing end to make it dividable
        minimum_interval = config.dispatcher_min_interval  # minimum keypress length
        sample_length = (44100 * config.dispatcher_window_size) / 1000  # size of individual sample = (100ms)


        # dict position:peak sum
        peaks = dict()
        for x in range(0, len(data) - 440):  # for every datapoint
            peaks[x] = np.sum(np.absolute(np.fft.fft(data[x:x + 440])))  # sum of absolute values returned by fft
            # values of fourier transform result of current 441 data points

        peaks = np.array(peaks.values())
        tau = np.percentile(peaks, config.dispatcher_threshold)  # 90% of peaks are below this number
        print("found " + str(sum(i > tau for i in peaks)) + " and have " + str(len(json)) + " key presses")
        events = []
        # todo: check delay of timestamp and press peak
        for timestamp, key in json:
            x = fromTimestamp(timestamp)
            if peaks[x] >= tau:  # peak must be higher than tau
                # if x - past_x >= minimum_interval:  # minimal distance between points/strokes
                    # It is a keypress event (maybe)
                keypress = al.normalize(data[x:x + sample_length])  # extract next 100ms
                past_x = x
                # Pass it immediately to workers
                #todo: get key number, pass it on
                out_queue.put([key, keypress])
                # Display event point
                # display_queue.put(x)
                events.append(keypress)

        display_queue.put(len(events))
        # todo: return txt file

        # Send termination flag to workers
    for _x in xrange(config.workers):
        out_queue.put((-1, None))

