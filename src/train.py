import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from model import Model1, get_cnn
from data import load_data
from config import Config

DATA_PATH = "data/PSI_Biology_solubility_trainset.csv"
SAVE_MODEL_PATH = "../models/"
LOAD_MODEL_PATH = "../model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_step = 0

y, x = load_data(DATA_PATH, Config.trim)

x = F.one_hot(torch.tensor(x).to(torch.int64), num_classes=21)
y = torch.tensor(y)
x = x.to(torch.float)
y = y.to(torch.float)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Initialisation

model = Model1()

# model.load_state_dict(torch.load(LOAD_MODEL_PATH))

optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = F.cross_entropy


# model.to(device)
# x_train.to(device)
# y_train.to(device)
# criterion.to(device)

# Training

model.train()
for e in range(Config.num_epochs):
    for i in range(0, x_train.size(0), Config.batch_size):
        optimiser.zero_grad()
        output = model(x_train[i : i + Config.batch_size]).squeeze()
        loss = criterion(output, y_train[i : i + Config.batch_size])
        loss.backward()
        optimiser.step()
        global_step += 1

    # Evaluation

    if global_step % Config.eval_step == 0:
        model.eval()
        loss = 0
        with torch.no_grad():
            for i in range(0, x_test.size(0), Config.batch_size):
                output = model(x_test[i : i + Config.batch_size])
                loss += criterion(output, y_test[i : i + Config.batch_size])

        print(f"Epoch {e} - Loss: {loss}")
        model.train()

        torch.save(model.state_dict(), SAVE_MODEL_PATH + "_" + str(e))
