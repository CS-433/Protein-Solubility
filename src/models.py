import torch.nn as nn

from config import Config


# CNN -> NN
class Model1(nn.Module):
    def __init__(self, params=Config.model1):
        super().__init__()

        self.cnn = nn.Sequential(*[ConvBlock(*param) for param in params["cnn"]])
        self.flatten = nn.Flatten()
        self.nn = nn.Sequential(*[LinearBlock(*param) for param in params["linear"]],)
        self.fc = nn.Linear(params["linear"][-1][1], 1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.nn(x)
        x = self.fc(x)

        return x


# CNN -> RNN -> NN
class Model2(nn.Module):
    def __init__(self, params=Config.model2):
        super().__init__()

        self.cnn = nn.Sequential(*[ConvBlock(*param) for param in params["cnn"]])

        self.flatten = nn.Flatten()
        self.rnn = nn.GRU(*params["rnn"], batch_first=True)
        self.fc = nn.Linear(params["rnn"][1] * Config.params["max_chain_length"], 1)

    def forward(self, x):
        x = self.cnn(x)

        # Get x to be the correct shape
        x = x.permute(0, 2, 1)
        x, h = self.rnn(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


# CNN -> NN
class Model3(nn.Module):
    def __init__(self, params=Config.model3):
        super().__init__()

        self.embed = EmbedLayer(*params["embed"])
        self.cnn = nn.Sequential(*[ConvBlock(*param) for param in params["cnn"]])
        self.flatten = nn.Flatten()
        self.nn = nn.Sequential(*[LinearBlock(*param) for param in params["linear"]],)
        self.fc = nn.Linear(params["linear"][-1][1], 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.nn(x)
        x = self.fc(x)

        return x


# Embed residues into lower dimensional space
class EmbedLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, x):
        seq_len = x.shape[2]

        x = x.permute(0, 2, 1).reshape(-1, self.dim_in)
        x = self.linear(x)
        x = x.reshape(-1, seq_len, self.dim_out).permute(0, 2, 1)

        return x


# For chaining GRU in a sequence
class SelectItem(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, inputs):
        return inputs[self.index]


# Convolution layer + Normalisation + Dropout + Non-linearity
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_p=0):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.bn = nn.BatchNorm1d(out_channels)

        if dropout_p != 0:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        if self.dropout != None:
            x = self.dropout(x)

        x = self.silu(x)

        return x


# Fully-Connected + Normalisation + Dropout + Non-Linearity
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_p=0):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        if dropout_p != 0:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)

        if self.dropout != None:
            x = self.dropout(x)

        x = self.silu(x)

        return x
