config = {
    "max_chain_length": 600,
    "num_epochs": 500,
    "eval_step": 10,
    "batch_size": 32,
    "weight_decay": 5e-3,
    "learning_rate": 1e-3
}

model_config = {
    "cnn": [
        # in_channels, out_channels, kernel_size, stride, padding
        [20, 4, 7, 1, 3],
        [4, 3, 5, 1, 2],
        [3, 2, 5, 1, 2]
    ],
    "clf": [
        # in_features, #out_features, bias
        [config["max_chain_length"]*2, config["max_chain_length"], True],
        [config["max_chain_length"], 1, True],
    ],
    "dropout_p": 0.2
}
