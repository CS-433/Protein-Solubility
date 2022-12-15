import torch.nn as nn

from config import Config


class PrintLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x


def conv1d_layer(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm1d(out_channels),
        nn.Dropout(p=Config.model["dropout_p"]),
        nn.SiLU(),
    )


def linear_layer(in_features, out_features, bias=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias),
        nn.BatchNorm1d(out_features),
        nn.Dropout(p=Config.model["dropout_p"]),
    )


def model1():
    return nn.Sequential(
        *[conv1d_layer(*param) for param in Config.model["cnn"]],
        nn.Flatten(),
        *[linear_layer(*param) for param in Config.model["linear"]],
    )


class model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(
            *Config.model["rnn"], batch_first=True
        )  # We can use LSTM/GRU/RNN
        self.seq = nn.Sequential(
            nn.Dropout(p=Config.model["dropout_p"]),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(Config.model["rnn"][1] * 20, 1),
            nn.BatchNorm1d(1),
            nn.Dropout(p=Config.model["dropout_p"]),
        )

    def forward(self, x):
        x = self.rnn(x)
        x = self.seq(x[0])
        return x


class model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq1 = nn.Sequential(
            *[conv1d_layer(*param) for param in Config.model["cnn"]],
        )
        self.rnn = nn.GRU(
            *Config.model["rnn"], batch_first=True
        )  # We can use LSTM/GRU/RNN
        self.seq2 = nn.Sequential(
            nn.Dropout(p=Config.model["dropout_p"]),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(Config.model["rnn"][1] * Config.model["cnn"][-1][1], 1),
            nn.BatchNorm1d(1),
            nn.Dropout(p=Config.model["dropout_p"]),
        )

    def forward(self, x):
        x = self.seq1(x)
        x = self.rnn(x)
        x = self.seq2(x[0])
        return x
