import json
import soundfile as sf


class Importer:

    @staticmethod
    def load(wav_file_path):
        wav_file, json_file = None, None

        try:
            wav_file = Importer.load_wav(wav_file_path)
        except:
            pass

        try:
            json_file = Importer.load_json(wav_file_path)
        except:
            pass

        return wav_file, json_file

    @staticmethod
    def load_wav(filename):
        mono, _ = sf.read(filename)
        return list(mono)

    @staticmethod
    def load_json(json_file_path):
        filename = json_file_path[:-3] + "json"
        return Importer.get_json(filename)

    @staticmethod
    def get_json(json_file_path):
        key_presses = dict()
        try:
            with open(json_file_path, 'r') as f:
                unicode_dict = json.load(f)
                for key in unicode_dict.keys():
                    key_presses[float(key)] = unicode_dict[key]
                return key_presses
        except Exception as e:
            print(key_presses, e)
            return None
