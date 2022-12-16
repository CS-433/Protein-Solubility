class Config:
    params = {
        "max_chain_length": 786, # longest sequence in dataset
        "num_epochs": 1000,
        "eval_step": 10,
        "batch_size": 32,
        "weight_decay": 5e-2,
        "learning_rate": 1e-3,
    }

    model1 = {
        "cnn": [
            # in_channels, out_channels, kernel_size, dropout_p
            [20, 2, 3, 0.2],
            [2, 2, 3, 0.2],
        ],
        "linear": [
            # in_features, #out_features, dropout_p
            [params["max_chain_length"] * 2, params["max_chain_length"], 0.2],
        ],
    }

    model2 = {
        "cnn": [
            # in_channels, out_channels, kernel_size, dropout_p
            [20, 4, 5, 0.2],
            [4, 3, 5, 0.2],
            [3, 2, 5, 0.2],
        ],
        # in, out, layers
        "rnn": [params["max_chain_length"], 300, 3],
        "dropout_p": 0.2
    }
