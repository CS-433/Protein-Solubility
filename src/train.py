import torch

from scores import print_scores


def train_epoch(data, model, optimiser, criterion, batch_size):
    model.train()
    for i in range(0, data["x_train"].shape[0], batch_size):
        optimiser.zero_grad()
        output = model(data["x_train"][i : i + batch_size]).squeeze()
        loss = criterion(output, data["y_train"][i : i + batch_size].float())
        loss.backward()
        optimiser.step()


def eval_model(epoch, data, model, optimiser, criterion):
    model.eval()
    with torch.no_grad():
        # Train loss
        output = model(data["x_train"]).squeeze()
        train_loss = criterion(output, data["y_train"].float())

        # Test loss
        y_pred = model(data["x_test"]).squeeze()
        test_loss = criterion(y_pred, data["y_test"].float())

        print("==========================================")
        print(
            f"Epoch {epoch} - Train loss: {train_loss:.3f};",
            f"Test loss: {test_loss:.3f}",
        )
        print_scores(y_pred, data["y_test"])
