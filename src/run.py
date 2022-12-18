import random

import numpy
import torch
import torch.nn as nn

from config import Config
from data import init_data
from models import *
from train import *


def setup_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # Ensure reproducibility
    setup_seed(42)

    DATA_PATH = "./data/PSI_Biology_solubility_trainset.csv"
    SAVE_MODEL_DIR = "./models/"
    SAVE_MODEL_PREFIX = "cnn_"
    LOAD_MODEL_PATH = None  # E.g. "./models/cnn_1"

    config = Config.params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = init_data(DATA_PATH, device, config)

    # Initialisation
    model = Model2()
    model.to(device)

    # Load pretrained weights
    if LOAD_MODEL_PATH != None:
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=data["neg_pos_ratio"]).to(device)

    # Training
    for e in range(1, config["num_epochs"] + 1):
        train_epoch(data, model, optimiser, criterion, config["batch_size"])

        # Evaluation
        if e % config["eval_step"] == 0:
            eval_model(e, data, model, optimiser, criterion)

            model.train()
            torch.save(model.state_dict(), SAVE_MODEL_DIR + SAVE_MODEL_PREFIX + str(e))


if __name__ == "__main__":
    main()
