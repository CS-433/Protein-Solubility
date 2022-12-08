import torch.nn as nn

from config import config, model_config


class PrintLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x


model = nn.Sequential(
    *sum([[nn.Conv1d(*sizes), nn.BatchNorm1d(sizes[1]), nn.Dropout(p=model_config["dropout_p"]), nn.SiLU()] for sizes in model_config["cnn"]], []),
    nn.Flatten(),
    *sum([[nn.Linear(*sizes), nn.BatchNorm1d(sizes[1]), nn.Dropout(p=model_config["dropout_p"])] for sizes in model_config["clf"]], [])
)
