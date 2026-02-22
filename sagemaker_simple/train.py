import numpy as np
import os
import sys
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import get_model


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")


def get_train_data():
    x_train = np.load(os.path.join(train_dir, "x_train.npy"))
    y_train = np.load(os.path.join(train_dir, "y_train.npy"))
    logger.info(f"x train: {x_train.shape}, y train: {y_train.shape}")
    return torch.from_numpy(x_train), torch.from_numpy(y_train)


def get_test_data():
    x_test = np.load(os.path.join(test_dir, "x_test.npy"))
    y_test = np.load(os.path.join(test_dir, "y_test.npy"))
    logger.info(f"x test: {x_test.shape}, y test: {y_test.shape}")
    return torch.from_numpy(x_test), torch.from_numpy(y_test)


def train(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train_data, y_train_data = get_train_data()
    x_test_data, y_test_data = get_test_data()

    batch_size = params.batchsize
    epochs = params.epochs
    learning_rate = params.lr
    logger.info(f"batch_size = {batch_size}, epochs = {epochs}, learning rate = {learning_rate}")

    train_dl = DataLoader(TensorDataset(x_train_data, y_train_data), batch_size, shuffle=True)
    test_dl = DataLoader(TensorDataset(x_test_data, y_test_data), batch_size)
    model = get_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for x_train, y_train in train_dl:
            x_batch = x_train.unsqueeze(1).to(device)
            y_batch = y_train.unsqueeze(1).to(device)
            y = model(x_batch)
            loss = criterion(y, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for x_test, y_test in test_dl:
                x_batch = x_test.unsqueeze(1).to(device)
                y_batch = y_test.unsqueeze(1).to(device)
                y = model(x_batch)
                loss = criterion(y, y_batch)
                total_loss += loss.item()
            mse = total_loss / len(train_dl)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Test Loss: {mse:.8f}")

    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_dir + "/sagemaker_test.pth")


if __name__ == "__main__":
    print("Running the training job ...")
    parser = argparse.ArgumentParser(description="Test Hyperparameters")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batchsize", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()
    print(f"Received hyperparameters: {args}")
    train(args)