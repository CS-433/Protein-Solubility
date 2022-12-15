class Config:
    params = {
        "max_chain_length": 600,
        "num_epochs": 1000,
        "eval_step": 10,
        "batch_size": 32,
        "weight_decay": 5e-2,
        "learning_rate": 1e-3,
    }

    model = {
        "cnn": [
            # in_channels, out_channels, kernel_size, stride, padding
            [20, 4, 5, 1, 2],
            [4, 3, 5, 1, 2],
            [3, 2, 5, 1, 2],
        ],
        # in, out, layers
        "rnn": [600, 300, 3],
        "linear": [
            # in_features, #out_features, bias
            [params["max_chain_length"] * 2, params["max_chain_length"], True],
            [params["max_chain_length"], 1, True],
        ],
        "dropout_p": 0.2,
    }
