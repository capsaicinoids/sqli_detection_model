#!/usr/bin/python

from configs.config import CFG
from models.model_1 import Model1

def exec():
    model = Model1(CFG)
    model.build_model()
    print(model.train_model())
if __name__ == '__main__':
    exec()
