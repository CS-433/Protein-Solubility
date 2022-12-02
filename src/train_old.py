import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from config import Config
from data import load_data
from model import Model1

DATA_PATH = "data/PSI_Biology_solubility_trainset.csv"
SAVE_MODEL_PATH = "models/"
LOAD_MODEL_PATH = "../model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_step = 0

y, x = load_data(DATA_PATH, Config.trim)

x = F.one_hot(torch.tensor(x).to(torch.int64), num_classes=21).to(device)
y = torch.tensor(y).to(device)

x = x.to(torch.float)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

y_train = y_train.to(torch.float)

# x_train = x_train[:0-x_train.size()[0]%Config.batch_size]
# y_train = y_train[:0-y_train.size()[0]%Config.batch_size]

# What is this? Train test split should do this?
x_test = x_test[: 0 - x_test.size()[0] % Config.batch_size]
y_test = y_test[: 0 - y_test.size()[0] % Config.batch_size]

# Initialisation

model = Model1()

# model.load_state_dict(torch.load(LOAD_MODEL_PATH))

optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
# Use nn, not F
criterion = F.cross_entropy

# .to is not inplace
model.to(device)
x_train.to(device)
y_train.to(device)

# Training
# torch.backends.cudnn.enabled should not need to be touched
torch.backends.cudnn.enabled = False
model.train()
for e in range(Config.num_epochs):
    for i in range(0, x_train.size(0), Config.batch_size):
        optimiser.zero_grad()
        output = model(x_train[i : i + Config.batch_size]).squeeze()
        loss = criterion(output, y_train[i : i + Config.batch_size].to(torch.long))
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
