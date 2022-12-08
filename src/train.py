import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from config import config
from data import encode_data, load_data
from models import model, model2
from scores import scores


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
model = model2()           #Comment this line for CNN model
model.to(device)
# Load pretrained weights
# model.load_state_dict(torch.load(LOAD_MODEL_PATH))

optimiser = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

# Set pos_weight w.r.t. distribution of y's
pos_weight = (y==0).sum()/y.sum()
#print(f"Positive sample weight: {pos_weight:.2f}")
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

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
        train_loss, test_loss, acc = 0, 0, 0
        with torch.no_grad():
            # Train loss
            #for i in range(0, x_train.shape[0], config["batch_size"]):
            #    output = model(x_train[i : i + config["batch_size"]]).squeeze()
            #    train_loss += criterion(
            #        output, y_train[i : i + config["batch_size"]].float()
            #    )
            #train_loss /= x_train.shape[0]
            output = model(x_train).squeeze()
            train_loss = criterion(output, y_train.float())

            # Test loss
            y_pred = model(x_test).squeeze()
            test_loss = criterion(y_pred, y_test.float())
            (acc, prec, rec, pred_std) = scores(y_pred, y_test)

        print(
            f"""Epoch {e + 1} - Train loss: {train_loss:.3f}; 
            Test loss: {test_loss:.3f}, Accuracy: {acc:.3f}, 
            Precision: {prec:.3f}, Recall: {rec:.3f}, Pred. STD: {pred_std:.3f}"""
        )

        model.train()

        torch.save(model.state_dict(), SAVE_MODEL_PATH + "cnn_" + str(e + 1))
