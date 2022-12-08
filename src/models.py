import torch.nn as nn

from config import model_config


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

class model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(*model_config["rnn"], batch_first=True)           #We can use LSTM/GRU/RNN
        self.seq = nn.Sequential(nn.Dropout(p=model_config["dropout_p"]), nn.SiLU(), nn.Flatten(),
                                 nn.Linear(model_config["rnn"][1] * 20, 1), nn.BatchNorm1d(1), nn.Dropout(p=model_config["dropout_p"]))

    def forward(self, x):
        x = self.rnn(x)
        x = self.seq(x[0])
        return x

class model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq1 = nn.Sequential(*sum([[nn.Conv1d(*sizes), nn.BatchNorm1d(sizes[1]), nn.Dropout(p=model_config["dropout_p"]), nn.SiLU()] for sizes in model_config["cnn"]], []))
        self.rnn = nn.GRU(*model_config["rnn"], batch_first=True)           #We can use LSTM/GRU/RNN
        self.seq2 = nn.Sequential(nn.Dropout(p=model_config["dropout_p"]), nn.SiLU(), nn.Flatten(),
                                  nn.Linear(model_config["rnn"][1] * model_config["cnn"][-1][1], 1), nn.BatchNorm1d(1), nn.Dropout(p=model_config["dropout_p"]))

    def forward(self, x):
        x = self.seq1(x)
        x = self.rnn(x)
        x = self.seq2(x[0])
        return x