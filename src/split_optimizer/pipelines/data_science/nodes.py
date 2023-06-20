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
from sklearn.metrics import ConfusionMatrixDisplay
from typing import Any, Dict, List, Tuple

# epochs: int, TRAINING_SIZE: int, dataset: list[np.ndarray]
from torch.utils.data.dataloader import DataLoader
import plotly.graph_objects as go

def train_model(
    epochs: int,
    TRAINING_SIZE: int,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
) -> Dict:
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

    return {"model": model, "model_history": model_history}


def test_model(
    model: nn.Module, loss_func: str, TEST_SIZE: int, test_dataloader: DataLoader
) -> Dict:
    model.eval()
    if loss_func == "MSELoss":
        calculate_loss = nn.MSELoss()

    with torch.no_grad():
        correct = 0
        test_loss = []
        predictions = []
        for data, target in test_dataloader:
            output = model(data)
            predictions.append(output)

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
            "pred": predictions,
        }

    return test_output


def plot_loss(model_history: dict, test_output: dict) -> plt.figure:
    
    epochs = range(1, len(list(model_history["train_loss_list"])) + 1)

    plt = go.Figure(
        [
            go.Scatter(
                x=list(epochs),
                y=model_history["train_loss_list"],
                mode='lines+markers',
                name="Training Loss"
            ),
            go.Scatter(
                x=list(epochs),
                y=model_history["val_loss_list"],
                mode='lines+markers',
                name="Validation Loss"
            )
        ]
    )
    plt.update_layout(
        title="Training and Validation Loss",
        xaxis_title="Epochs",
        yaxis_title="Loss"
    )

    return plt



def plot_confusionmatrix(test_output: dict, test_dataloader: DataLoader):
    predictions_onehot = test_output["pred"]
    test_features, test_labels_onehot = next(iter(test_dataloader))
    # convert one-hot encoding to label-encoding
    predictions = []
    for i in predictions_onehot[0]:
        predictions.append(np.argmax(i).item())

    test_labels = []
    for i in test_labels_onehot:
        test_labels.append(np.argmax(i).item())

    # calculate and display confusionmatrix
    ConfusionMatrixDisplay.from_predictions(test_labels, predictions, cmap="OrRd")

    plt.title(f"Confusionmatrix")

    return plt
