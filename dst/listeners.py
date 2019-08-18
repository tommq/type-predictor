import json

import utils
from dst.libraries import al
import warnings

warnings.filterwarnings("ignore")


def wavfile(in_file, out_queue, config):
    _, mono = al.load(in_file)
    out_queue.put(list(mono))
    out_queue.put(None)


def jsonfile(json_file, out_queue, config):
    filename = json_file[:-3] + "json"
    out_queue.put(get_json(filename))
    out_queue.put(None)


def get_json(json_file):
    key_presses = dict()
    try:
        with open(json_file) as f:
            unicode_dict = json.load(f)

            for key in unicode_dict.keys():
                key_presses[utils.dt_from_str(str(key))] = str(unicode_dict[key])

            return key_presses
    except Exception as e:
        print(key_presses, e)


def input_recording():
    pass


def input_interactive():
    pass
