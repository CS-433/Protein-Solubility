class Config:
    params = {
        "max_chain_length": 810,  # Sequence trim length
        "num_epochs": 1000, # Number of iterations
        "eval_step": 10, # Evaluate model each ... iters
        "batch_size": 32,
        "weight_decay": 1e-2,
        "learning_rate": 1e-3,
    }

    # Model 1: CNN + NN
    model1 = {
        "cnn": [
            # in_channels, out_channels, kernel_size, dropout_p
            [20, 4, 3, 0.25],
            [4, 2, 3, 0.25],
            [2, 1, 3, 0.25],
        ],
        "linear": [
            # in_features, #out_features, dropout_p
            [params["max_chain_length"], params["max_chain_length"]//2, 0.25],
        ],
    }

    # Model 2: CNN + RNN + NN
    model2 = {
        "cnn": [
            # in_channels, out_channels, kernel_size, dropout_p
            [20, 4, 7, 0.2],
            [4, 3, 5, 0.2],
            [3, 1, 5, 0.2],
        ],
        # input_size, hidden_size, num_layers
        "rnn": [1, 2, 3],
    }

    # Model 2: EMBED + CNN + NN
    model3 = {
        "embed": [20, 2],
        "cnn": [
            # in_channels, out_channels, kernel_size, dropout_p
            [2, 2, 3, 0.2],
            [2, 2, 3, 0.2],
            [2, 2, 3, 0.2],
        ],
        "linear": [
            # in_features, #out_features, dropout_p
            [params["max_chain_length"] * 2, params["max_chain_length"], 0.2],
        ],
    }
