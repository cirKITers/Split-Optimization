from .hybrid_model import Net

import torch
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, List, Tuple

# epochs: int, TRAINING_SIZE: int, dataset: list[np.ndarray]
from torch.utils.data.dataloader import DataLoader

def train_model(
    epochs: int, TRAINING_SIZE: int, train_dataloader:DataLoader, test_dataloader:DataLoader
) ->Dict:
    model = Net()
    loss_func = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss_list = []
    val_loss_list = []
    for epoch in range(epochs):
        total_loss = []
        for batch_idx, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())

        train_loss_list.append(sum(total_loss) / len(total_loss))
        print(
            "Training [{:.0f}%]\tLoss: {:.4f}".format(
                100.0 * (epoch + 1) / epochs, train_loss_list[-1]
            )
        )

        model.eval()
        with torch.no_grad():
            epoch_loss = []
            for data, target in test_dataloader:
                output = model(data)
                loss = loss_func(output, target)

                epoch_loss.append(loss.item())

        val_loss_list.append(np.mean(epoch_loss))

    model_history = {
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list,
        "loss_func": loss_func,
    }

    return {"model":model, "model_history":model_history}


def test_model(model: nn.Module, loss_func: str, TEST_SIZE: int, test_dataloader:DataLoader)->Dict:
    model.eval()
    if loss_func == "MSELoss":
        calculate_loss = nn.MSELoss()

    with torch.no_grad():
        correct = 0
        test_loss = []
        for data, target in test_dataloader:
            output = model(data)

            pred = output.argmax()
            if pred == target.argmax():
                correct += 1

            loss = calculate_loss(output, target)
            test_loss.append(loss.item())

            accuracy = correct / TEST_SIZE * 100
            average_test_loss = sum(test_loss) / len(test_loss)

        print(
            "Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
                average_test_loss, accuracy
            )
        )
        test_output = {
            "average_test_loss": average_test_loss,
            "accuracy": accuracy,
            "pred": pred,
        }

    return test_output


def plot_loss(model_history: dict, test_output: dict)->plt.figure:
    fig = plt.figure()
    plt.plot(model_history["train_loss_list"])
    plt.title("Hybrid NN Training Convergence")
    plt.xlabel("Training Iterations")
    plt.ylabel("Neg Log Likelihood Loss")
    plt.savefig("plot.png")
    return fig


def plot_result_picture(test_output: dict):
    result_picture = 0
    return result_picture


def plot_confusionmatrix(test_output: dict, test_dataloader: DataLoader):
    confusionmatrix = 0
    return confusionmatrix
