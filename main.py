#!/usr/bin/python

from configs.config import CFG
from models.model_1 import Model1

def exec():
    model = Model1(CFG)
    model.load_data()
    print(model.get_data())

if __name__ == '__main__':
    exec()
