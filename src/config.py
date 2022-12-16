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
            # in_channels, out_channels, kernel_size, stride, padding
            [20, 4, 5, 1, 2],
            [4, 3, 5, 1, 2],
            [3, 2, 5, 1, 2],
        ],
        "linear": [
            # in_features, #out_features, bias
            [params["max_chain_length"] * 2, params["max_chain_length"]],
            [params["max_chain_length"], 1],
        ],
        "dropout_p": 0.2,
    }

    model2 = {
        "cnn": [
            # in_channels, out_channels, kernel_size, stride, padding
            [20, 4, 5, 1, 2],
            [4, 3, 5, 1, 2],
            [3, 2, 5, 1, 2],
        ],
        # in, out, layers
        "rnn": [params["max_chain_length"], 300, 3],
        "dropout_p": 0.2
    }
