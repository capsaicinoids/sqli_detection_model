#!/usr/bin/python

from configs.config import CFG
from models.model_1 import Model1

def exec():
    model = Model1(CFG)
    model.build_model()
    model.train_model()
    model.evaluate_model()
if __name__ == '__main__':
    exec()
