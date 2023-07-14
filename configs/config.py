"""
    Model Config in JSON format
"""

CFG: dict = {
    "data": {
        "path": "dataloader/wrangling/processed_datasets/cleaned_sqli.csv",
        "batch_size": 32,
        "shuffle": True,
        "buffer_size": None,
        "train_size": 0.8,
        "tokenization": {
            "vocab_size": 5000,
            "embedding_dim": 128,
            "max_length": 400,
            "trunc_type": 'post',
            "padding_type": 'post',
            "oov_tok": "<OOV>"
        }
    },
    "train": {
        "optimizer": {
            "type": 'adam',
            "learning_rate": 0.001
        },
        "metrics": ["accuracy"],
        "loss": "binary_crossentropy",
        "epochs": 30,
        "validation_split": 0.2
    },
    "model": {
        "dense_layers": {
            "layer_1": {
                "units": 64,
                "activation": "relu",
                "name": "layer_1",
            },
            "layer_2": {
                "units": 32,
                "activation": "relu",
                "name": "layer_2",
            },
            "layer_3": {
                "units": 16,
                "activation": "relu",
                "name": "layer_3",
            },
            "layer_4": {
                "units": 1,
                "activation": "sigmoid",
                "name": "output_layer",
            }
        }
    }
}


# Metrics = [confusion_matrix, accuracy, precision, recall, f1_score]