import torch.nn as nn

class Model1(nn.Module): 
    def __init__(self, ): 
        super().__init__()
        self.c1 = nn.Conv2d(1, 10, 5, padding=2)
        self.r1 = nn.ReLU()
        self.p1 = nn.MaxPool2d(2,2)
        self.c2 = nn.Conv2d(10, 20, 5, padding=2)
        self.r2 = nn.ReLU()
        self.p2 = nn.AdaptiveAvgPool2d(1)
        self.c3 = nn.Conv2d(20, 40, 5, padding=2)
        self.r3 = nn.ReLU()
        self.p3 = nn.MaxPool2d(2,2)
        self.f = nn.Flatten()
        self.fc1 = nn.Linear(100, 100) #TODO: fix size
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)
        
    def forward(self, Z): 
        Z = self.c1(Z)
        Z = self.r1(Z)
        Z = self.p1(Z)
        Z = self.c1(Z)
        Z = self.r1(Z)
        Z = self.p1(Z)
        Z = self.c1(Z)
        Z = self.r1(Z)
        Z = self.p1(Z)
        Z = self.f(Z)
        Z = self.fc1(Z)
        Z = self.fc2(Z)
        Z = self.fc3(Z)
        return Z

def get_cnn():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 5, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(64, 128, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(128, 10, 1),
        nn.Flatten(),
    )