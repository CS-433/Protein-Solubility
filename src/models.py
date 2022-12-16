import torch.nn as nn

from config import Config

def model1(params = Config.model1):
    return nn.Sequential(
        *[ConvBlock(*param) for param in params["cnn"]],
        nn.Flatten(),
        *[LinearBlock(*param) for param in params["linear"]],
        nn.Linear(params["linear"][-1][1], 1)
    )

def model2(params = Config.model2):
    return nn.Sequential(
        *[ConvBlock(*param) for param in params["cnn"]],
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

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_p = 0):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.bn = nn.BatchNorm1d(out_channels)
        
        if (dropout_p != 0):
            self.dropout = nn.Dropout(p = dropout_p)
        else:
            self.dropout = None

        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        
        if (self.dropout != None):
            x = self.dropout(x)
        
        x = self.silu(x)

        return x

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_p = 0):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        if (dropout_p != 0):
            self.dropout = nn.Dropout(p = dropout_p)
        else:
            self.dropout = None

        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        
        if (self.dropout != None):
            x = self.dropout(x)
        
        x = self.silu(x)

        return x
