import numpy as np
import pandas as pd

class Preprocessing():
    @staticmethod
    def preprocessing(text):
        text = str(text).lower()
        return text

    @staticmethod
    def load_data(data_config):
        data = pd.read_csv(data_config.path)
        data['Label'] = data['Label'].apply(Preprocessing.preprocessing)

        return data
