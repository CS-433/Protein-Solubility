import torch.nn as nn

from config import Config

def model1(params = Config.model1):
    return nn.Sequential(
        *[conv1d_seq(*param, params["dropout_p"]) for param in params["cnn"]],
        nn.Flatten(),
        *[linear_seq(*param, params["dropout_p"]) for param in params["linear"]],
    )

def model2(params = Config.model2):
    return nn.Sequential(
        *[conv1d_seq(*param, params["dropout_p"]) for param in params["cnn"]],
        nn.GRU(
            *params["rnn"], batch_first=True
        ),
        SelectItem(0),
        nn.Dropout(p=params["dropout_p"]),
        nn.SiLU(),
        nn.Flatten(),
        nn.Linear(params["rnn"][1] * params["cnn"][-1][1], 1),
    )

class PrintLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x

class SelectItem(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, inputs):
        return inputs[self.index]


def conv1d_seq(in_channels, out_channels, kernel_size, stride, padding, dropout_p):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm1d(out_channels),
        nn.Dropout(p=dropout_p),
        nn.SiLU(),
    )


def linear_seq(in_features, out_features, dropout_p):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.Dropout(p=dropout_p),
    )
