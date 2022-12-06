config = {
    "max_chain_length": 500,
    "num_epochs": 100,
    "eval_step": 1,
    "batch_size": 32,
}

model_config = {
    "cnn": [
        # in_channels, out_channels, kernel_size, stride, padding
        [20, 16, 5, 1, 2],
        [16, 8, 5, 1, 2],
        [8, 4, 5, 1, 2],
    ],
    "clf": [config["max_chain_length"] * 4, 1],
    "dropout_p": 0.3
}
