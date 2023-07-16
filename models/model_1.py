from .base_model import BaseModel
from dataloader.preprocessing.preprocessing import Preprocessing

import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class Model1(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.dataset = Preprocessing().load_data(self.config.data)
        self.model = None
        self.tokenizer = None
    
    def load_data(self):
        """
        Load and prepare the data for training and testing.

        Returns:
            X_train (ndarray): The training data.
            X_test (ndarray): The testing data.
            y_train (ndarray): The training labels.
            y_test (ndarray): The testing labels.
        """

        dataset = self.dataset

        # query = dataset['Sentence'].values
        # labels = dataset['Label'].values
        
        # Take 20% of the data
        sample_size = int(len(dataset) * 0.2)
        query = dataset.iloc[:sample_size]['Sentence'].values
        labels = dataset.iloc[:sample_size]['Label'].values

        X_train, X_test, y_train, y_test = train_test_split(query, labels, train_size=self.config.data.train_size, random_state=42, shuffle=self.config.data.shuffle)

        return X_train, X_test, y_train, y_test
    
    def tokenize_data(self):
        """
        Tokenizes the data by converting the text queries into sequences of integers using a tokenizer. 

        Returns:
            train_padded (numpy.ndarray): The padded sequences of integers representing the training queries.
            training_labels (list): The labels corresponding to the training queries.
            validation_padded (numpy.ndarray): The padded sequences of integers representing the validation queries.
            validation_labels (list): The labels corresponding to the validation queries.
        """
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
        """
        Builds and initializes a Keras Sequential model.

        This function creates a Keras Sequential model by stacking various layers.
        The layers include an Embedding layer, Dropout layers, a GlobalAveragePooling1D layer, and a Dense layer.
        The model is then assigned to the 'model' attribute of the current object.

        Parameters:
            None

        Returns:
            None
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.config.data.tokenization.vocab_size , self.config.data.tokenization.embedding_dim, input_length=self.config.data.tokenization.max_length),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model = model

    def train_model(self):
        """
        Train the model using the tokenized data and return the training and validation loss.

        Parameters:
            None

        Returns:
            A tuple containing the training and validation loss as lists.
        """
        train_padded, training_labels, validation_padded, validation_labels = self.tokenize_data()

        self.model.compile(optimizer=self.config.train.optimizer.type, loss=self.config.train.loss, metrics=self.config.train.metrics)
        model_history = self.model.fit(
            train_padded, training_labels, epochs=self.config.train.epochs, 
            validation_data=(validation_padded, validation_labels), validation_split=self.config.train.validation_split, 
            verbose=self.config.train.verbose, steps_per_epoch=self.config.train.steps_per_epoch
        )

        return model_history.history['loss'], model_history.history['val_loss']

    def evaluate_model(self):
        if self.config.model.save_model_path:
            self.model.save(self.config.model.save_model_path)
        
        if self.config.model.save_tokenization_path:
            with open(self.config.model.save_tokenization_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)