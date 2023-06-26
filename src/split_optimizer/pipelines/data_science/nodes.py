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
    loss_func: str,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
) -> Dict:
    

    model = Net()
    if loss_func == "MSELoss":
        calculate_loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss_list = []
    val_loss_list = []
    for epoch in range(epochs):
        total_loss = []
        for batch_idx, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = calculate_loss(output, target)
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
                loss = calculate_loss(output, target)

                epoch_loss.append(loss.item())

        val_loss_list.append(np.mean(epoch_loss))

    model_history = {"train_loss_list": train_loss_list, "val_loss_list": val_loss_list}
    model_tracking = model_history
    
    return {"model": model, "model_history": model_history, "model_tracking":model_tracking}


def test_model(
    model: nn.Module, loss_func: str, TEST_SIZE: int, test_dataloader: DataLoader
) -> Dict:
    model.eval()
    if loss_func == "MSELoss":
        calculate_loss = nn.MSELoss()

    with torch.no_grad():
        correct = 0
        test_loss = []
        predictions_onehot = []
        for data, target in test_dataloader:
            output = model(data)
            predictions_onehot.append(output)

            for i in output:
                pred = i.argmax()
                if pred == target.argmax():
                    correct += 1 


            loss = calculate_loss(output, target)
            test_loss.append(loss.item())

        accuracy = correct / TEST_SIZE 
        average_test_loss = sum(test_loss) / len(test_loss)

        print(
            "Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}".format(
                average_test_loss, accuracy
            )
        )

        label_predictions = []
        for i in predictions_onehot:
            label_predictions.append(np.argmax(i).item())

        test_output = {
            "average_test_loss": average_test_loss,
            "accuracy": accuracy,
            "pred": label_predictions,
        }
        test_tracking = test_output

    return test_output, test_tracking


def plot_loss(model_history: dict) -> plt.figure:

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
    
    test_labels_onehot=[]
    for _, target in test_dataloader:
        test_labels_onehot.append(target)
    
    label_predictions = test_output["pred"]
    
    test_labels = []
    for i in test_labels_onehot:
        test_labels.append(np.argmax(i).item())

    confusion_matrix = metrics.confusion_matrix(test_labels, label_predictions)
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
        xaxis_title="Real Label",
        yaxis_title="Predicted Label",
    )
    return fig

def parameter_tracking(
        epochs:int,
        learning_rate:float,
        loss_func:str,
        TRAINING_SIZE:int,
        TEST_SIZE:int,
        number_of_qubits:int
):
    params_tracking = {
        "epochs":epochs,
        "learning_rate":learning_rate,
        "loss_func":loss_func,
        "training_size":TRAINING_SIZE,
        "test_size":TEST_SIZE,
        "number_of_qubits":number_of_qubits
    }
    return params_tracking
