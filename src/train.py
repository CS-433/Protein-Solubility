import torch
import torch.nn as nn
from pandas import read_csv
from sklearn.model_selection import train_test_split

from config import Config


DATA_PATH = "../data/PSI_Biology_solubility_trainset.csv"
SAVE_MODEL_PATH = "../models/"
LOAD_MODEL_PATH = "../model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_step = 0

# Dataset (to be put in data.py)

data = read_csv(DATA_PATH)

y = data.solubility.values
x = data.fasta.values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Initialisation

model = ...  # model.to(device)
# model.load_state_dict(torch.load(LOAD_MODEL_PATH))
optimiser = torch.optim.AdamW(model.params(), lr=1e-3)
criterion = nn.BCELoss()

# Training

model.train()
for e in range(Config.num_epochs):
    for i in range(0, x_train.size(0), Config.batch_size):
        optimiser.zero_grad()
        output = model(x_train[i : i + Config.batch_size])
        loss = criterion(output, y_train[i : i + Config.batch_size])
        loss.backward()
        optimiser.step()
        global_step += 1

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
