from tensorflow.keras.optimizers import Adam, SGD, RMSprop

"""
    Model Config in JSON format
"""

CFG: dict = {
    "data": {
        "path": "dataloader/wrangling/processed_datasets/cleaned_sqli.csv",
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
        "validation_split": 0.2
    },
    "model": {
        "input_shape": (28, 28, 1),
        "dense_layers": {
            "layer_1": {
                "type": "Dense",
                "units": 64,
                "activation": "relu",
                "name": "layer_1",
            },
            "layer_2": {
                "type": "Dense",
                "units": 32,
                "activation": "relu",
                "name": "layer_2",
            },
            "layer_3": {
                "type": "Dense",
                "units": 16,
                "activation": "relu",
                "name": "layer_3",
            },
            "layer_4": {
                "type": "Dense",
                "units": 1,
                "activation": "sigmoid",
                "name": "output_layer",
            }
        }
    }
}


# Metrics = [confusion_matrix, accuracy, precision, recall, f1_score]