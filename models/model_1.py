from .base_model import BaseModel
from dataloader.preprocessing.preprocessing import Preprocessing

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class Model1(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.dataset = Preprocessing().load_data(self.config.data)
        self.tokenizer = None
    
    def load_data(self):
        dataset = self.dataset
        query = dataset['Sentence'].values
        labels = dataset['Label'].values

        X_train, X_test, y_train, y_test = train_test_split(query, labels, train_size=self.config.data.train_size, random_state=42, shuffle=self.config.data.shuffle)

        return X_train, X_test, y_train, y_test
    
    def tokenize_data(self):
        training_query, training_labels = self.load_data()[0], self.load_data()[2]
        validation_query, validation_labels = self.load_data()[1], self.load_data()[3]

        tokenizer = Tokenizer(
            num_words=self.config.data.tokenization.vocab_size,
            oov_token=self.config.data.tokenization.oov_tok)
        tokenizer.fit_on_texts(training_query)

        train_seq = tokenizer.texts_to_sequences(training_query)
        validation_seq = tokenizer.texts_to_sequences(validation_query)

        self.tokenizer = tokenizer

        train_padded = pad_sequences(train_seq, maxlen=self.config.data.tokenization.max_length)
        validation_padded = pad_sequences(validation_seq, maxlen=self.config.data.tokenization.max_length)

        return train_padded, training_labels, validation_padded, validation_labels

     
    def build_model(self):
        pass