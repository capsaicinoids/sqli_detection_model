from abc import ABC, abstractmethod
from utils.utils import Config

class BaseModel(ABC):
    # This code act as a base model for all child models (Inheritance)
    def __init__(self, config):
        self.config = Config.from_json(config)
    
    # An abstract method is a method that is declared in a base class but does not have an implementation. 
    # It helps enforce the structure and behavior of subclasses by requiring them to implement specific methods defined in the abstract base class 

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass