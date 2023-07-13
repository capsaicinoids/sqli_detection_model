from .base_model import BaseModel
from dataloader.preprocessing.preprocessing import Preprocessing

import tensorflow as tf
from tensorflow.keras.layers import Dense

class Model1(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.dataset = None
    
    def load_data(self):
        self.dataset = Preprocessing().load_data(self.config.data) 

    def get_data(self):
        return self.dataset
    
    def build_model(self):
        pass