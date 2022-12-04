import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from config import config
from data import encode_data, load_data
from models import model

DATA_PATH = "./data/PSI_Biology_solubility_trainset.csv"
SAVE_MODEL_PATH = "./models/"
LOAD_MODEL_PATH = "./model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
y, x = load_data(DATA_PATH, config["max_chain_length"])
y, x = encode_data(y, x, config["max_chain_length"])

x = x.to(device)
y = y.to(device)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Initialisation

model.to(device)
# Load pretrained weights
# model.load_state_dict(torch.load(LOAD_MODEL_PATH))

optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss().to(device)

# Training

model.train()
for e in range(config["num_epochs"]):
    for i in range(0, x_train.shape[0], config["batch_size"]):
        optimiser.zero_grad()
        output = model(x_train[i : i + config["batch_size"]]).squeeze()
        loss = criterion(output, y_train[i : i + config["batch_size"]].float())
        loss.backward()
        optimiser.step()

    # Evaluation
    if e % config["eval_step"] == config["eval_step"] - 1:
        model.eval()
        train_loss, test_loss = 0, 0
        with torch.no_grad():
            for i in range(0, x_train.shape[0], config["batch_size"]):
                output = model(x_train[i : i + config["batch_size"]]).squeeze()
                train_loss += criterion(
                    output, y_train[i : i + config["batch_size"]].float()
                )

            for i in range(0, x_test.shape[0], config["batch_size"]):
                output = model(x_test[i : i + config["batch_size"]]).squeeze()
                test_loss += criterion(
                    output, y_test[i : i + config["batch_size"]].float()
                )

        print(
            f"Epoch {e + 1} - Train loss: {train_loss / y_train.shape[0]}, Test loss: {test_loss / y_test.shape[0]}"
        )
        model.train()

        torch.save(model.state_dict(), SAVE_MODEL_PATH + "_" + str(e))
