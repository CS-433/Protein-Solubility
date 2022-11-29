import torch.nn as nn

from config import Config


## 3 Conv layers, each followd by a Maxpool layer -> flatten -> 3 Linear layers -> y
class Model1(nn.Module):
    def __init__(self,):
        super().__init__()
        self.c1 = nn.Conv2d(Config.batch_size, Config.batch_size, 5, padding=2)
        self.r1 = nn.ReLU()
        self.p1 = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(Config.batch_size, Config.batch_size, 5, padding=2)
        self.r2 = nn.ReLU()
        self.p2= nn.MaxPool2d(3,2)
        self.c3 = nn.Conv2d(Config.batch_size, Config.batch_size, 5, padding=2)
        self.r3 = nn.ReLU()
        self.p3 = nn.MaxPool2d(2, 2)
        self.f = nn.Flatten()
        self.fc1 = nn.Linear(124, 124)
        self.fc2 = nn.Linear(124, 124)
        self.fc3 = nn.Linear(124, 1)

    def forward(self, Z):
        Z = self.c1(Z)
        Z = self.r1(Z)
        Z = self.p1(Z)
        Z = self.c2(Z)
        Z = self.r2(Z)
        Z = self.p2(Z)
        Z = self.c3(Z)
        Z = self.r3(Z)
        Z = self.p3(Z)
        Z = self.f(Z)
        Z = self.fc1(Z)
        Z = self.fc2(Z)
        Z = self.fc3(Z)
        return Z
