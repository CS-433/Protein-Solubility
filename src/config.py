config = {
    "num_epochs": 1000,
    "batch_size": 20,
    "eval_step": 100,
    "max_chain_length": 500,
}

model_config = {
    "cnn": [
        # in_channels, out_channels, kernel_size, stride, padding
        [20, 16, 5, 1, 2],
        [16, 8, 5, 1, 2],
        [8, 4, 5, 1, 2],
    ],
    "clf": [config["max_chain_length"] * 4, 1],
}
