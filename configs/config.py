from tensorflow.keras.optimizers import Adam, SGD, RMSProp

"""
    Model Config in JSON format
"""

CFG = {
    "data": {
        "path": "",
        "batch_size": 32,
        "shuffle": True,
        "buffer_size": None,
        "train_size": 0.8
    },
    "train": {
        "optimizer": {
            "type": Adam(),
        },
        "metrics": ["accuracy"],
        "loss": "binary_crossentropy",
        "epochs": 30,
    },
    "model": {
        "input_shape": (28, 28, 1),
        "dense_layers": {
            "layer_1": {
                "type": "Dense",
                "units": 64,
                "activation": "relu",
                "name": "layer_1",
            }
        }
    }
}