import torch.nn as nn

from config import *


model = nn.Sequential(
    *sum([[nn.Conv1d(*sizes), nn.SiLU()] for sizes in model_config], []),
    nn.Linear(model_config["clf"], 1)
)
