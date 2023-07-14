#!/usr/bin/python

from configs.config import CFG
from models.model_1 import Model1

def exec():
    model = Model1(CFG)
    print(model.tokenize_data()[0])
if __name__ == '__main__':
    exec()
