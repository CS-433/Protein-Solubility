config = {
    "max_chain_length": 600,
    "num_epochs": 200,
    "eval_step": 10,
    "batch_size": 32,
    "L2_reg": 3e-3,
    "lr": 1e-3
}

model_config = {
    "cnn": [
        # in_channels, out_channels, kernel_size, stride, padding
        [20, 16, 5, 1, 2],
        [16, 8, 5, 1, 2],
        [8, 4, 5, 1, 2],
    ],
    "clf": [
        [config["max_chain_length"] * 4, config["max_chain_length"] * 2],
        [config["max_chain_length"] * 2, config["max_chain_length"]],
        [config["max_chain_length"], 1]
    ],
    "dropout_p": 0.3
}
