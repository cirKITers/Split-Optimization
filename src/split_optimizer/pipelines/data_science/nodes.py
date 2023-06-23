from .hybrid_model import Net

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn import metrics
import plotly.figure_factory as ff
from typing import Any, Dict, List, Tuple
import plotly.express as px


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
                mode="lines+markers",
                name="Training Loss",
            ),
            go.Scatter(
                x=list(epochs),
                y=model_history["val_loss_list"],
                mode="lines+markers",
                name="Validation Loss",
            ),
        ]
    )
    plt.update_layout(
        title="Training and Validation Loss", xaxis_title="Epochs", yaxis_title="Loss"
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

    confusion_matrix = metrics.confusion_matrix(test_labels, predictions)
    confusion_matrix = confusion_matrix.transpose()
    labels = ["0", "1", "3", "6"]
    fig = px.imshow(
        confusion_matrix,
        x=labels,
        y=labels,
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    z_text = z_text = [[str(y) for y in x] for x in confusion_matrix]
    fig.update_traces(text=z_text, texttemplate="%{text}")
    fig.update_layout(
     title_text="Confusion Matrix",
     xaxis_title = "Real Label",
     yaxis_title = "Predicted Label")
    fig.update_xaxes()
    fig.show()
    return fig
