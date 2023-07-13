import pandas as pd

class Preprocessing():
    @staticmethod
    def preprocessing(text):
        text = str(text).lower()
        return text

    @staticmethod
    def load_data(data_config):
        data = pd.read_csv(data_config.path)
        data['Sentence'] = data['Sentence'].apply(Preprocessing.preprocessing)

        return data
