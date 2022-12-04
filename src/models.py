import torch.nn as nn

from config import model_config


model = nn.Sequential(
    *sum([[nn.Conv1d(*sizes), nn.SiLU()] for sizes in model_config["cnn"]], []),
    nn.Linear(*model_config["clf"])
)
