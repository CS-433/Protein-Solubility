import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from config import config
from models import model
from data import load_data, encode_data

DATA_PATH = "data/PSI_Biology_solubility_trainset.csv"
SAVE_MODEL_PATH = "../models/"
LOAD_MODEL_PATH = "../model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_step = 0

# Load data
# x.size() should be equal to [num_samples, 20 (num channels), 500 (length of sequences)]
# y is an array of labels
y, x = load_data(DATA_PATH, config['max_chain_length'])
y, x = encode_data(y, x, config['max_chain_length'])

# move data to GPU
# x = x.to(device)
# y = y.to(device)

if device == torch.device('cuda'):
    x = x.to(device)
    y = y.to(device)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Initialisation

model.to(device)
# Load pretrained weights
# model.load_state_dict(torch.load(LOAD_MODEL_PATH))

optimiser = torch.optim.AdamW(model.params(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss().to(device)

# Training

model.train()
for e in range(config["num_epochs"]):
    for i in range(0, x_train.size(0), config["batch_size"]):
        optimiser.zero_grad()
        output = model(x_train[i : i + config["batch_size"]])
        loss = criterion(output, y_train[i : i + config["batch_size"]])
        loss.backward()
        optimiser.step()
        global_step += 1

    # Evaluation
    if global_step % config["eval_step"] == 0:
        model.eval()
        train_loss, test_loss = 0, 0
        with torch.no_grad():
            for i in range(0, x_train.size(0), config["batch_size"]):
                output = model(x_train[i : i + config["batch_size"]])
                train_loss += criterion(output, y_train[i : i + config["batch_size"]])

            for i in range(0, x_test.size(0), config["batch_size"]):
                output = model(x_test[i : i + config["batch_size"]])
                test_loss += criterion(output, y_test[i : i + config["batch_size"]])

        print(f"Epoch {e + 1} - Train loss: {train_loss}, Test loss: {test_loss}")
        model.train()

        torch.save(model.state_dict(), SAVE_MODEL_PATH + "_" + str(e))
