from .hybrid_model import Net

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import metrics
from typing import Dict, List
import plotly.express as px
import mlflow

from torch.utils.data.dataloader import DataLoader
import plotly.graph_objects as go

from .instructor import Instructor

def train_model(
    instructor: Instructor,
    epochs: int,
) -> Dict:
    train_loss_list = []
    val_loss_list = []
    for epoch in range(epochs):
        instructor.model.train()
        train_loss = []
        for data, target in instructor.train_dataloader:
            loss = instructor.objective_function(data=data, target=target)

            instructor.optimizer.zero_grad()
            loss.backward()
            instructor.optimizer.step(data, target, instructor.objective_function)
            train_loss.append(loss.item())

        train_loss_list.append(np.mean(train_loss))
        print(
            "Training [{:.0f}%]\tLoss: {:.4f}".format(
                100.0 * (epoch + 1) / epochs, train_loss_list[-1]
            )
        )

        instructor.model.eval()
        with torch.no_grad():
            val_loss = []
            for data, target in instructor.test_dataloader:
                loss = instructor.objective_function(
                    data=data, target=target, train=False
                )

                val_loss.append(loss.item())

        val_loss_list.append(np.mean(val_loss))

    model_history = {"train_loss_list": train_loss_list, "val_loss_list": val_loss_list}

    return {
        "model": instructor.model,
        "model_history": model_history,
    }


def test_model(
    instructor: Instructor,
    model: nn.Module,
) -> Dict:
    instructor.model.eval()

    with torch.no_grad():
        correct = 0
        test_loss_list = []
        predictions = []
        for data, target in instructor.test_dataloader:
            output = model(data)

            predictions.append(output)

            for i in output:
                pred = i.argmax()
                if pred == target.argmax():
                    correct += 1

            loss = instructor.test_loss(output, target)
            test_loss_list.append(loss.item())

        accuracy = correct / len(test_loss_list)
        average_test_loss = sum(test_loss_list) / len(test_loss_list)

        print(
            "Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}".format(
                average_test_loss, accuracy
            )
        )

        label_predictions = []
        for i in predictions:
            label_predictions.append(np.argmax(i).item())

        test_output = {
            "average_test_loss": average_test_loss,
            "accuracy": accuracy,
            "pred": label_predictions,
        }

    return {"test_output": test_output}


def create_instructor(
    model: nn.Module,
    loss_func: str,
    learning_rate: float,
    optimizer_list: List,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    class_weights_train: List,
):
    instructor = Instructor(
        model=model,
        loss_func=loss_func,
        learning_rate=learning_rate,
        optimizer_list=optimizer_list,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        class_weights_train=class_weights_train,
    )

    return {"instructor": instructor}


def create_model(n_qubits: int, n_layers: int, classes: List):
    model = Net(n_qubits=n_qubits, n_layers=n_layers, classes=classes)

    return {"model": model}


def mlflow_tracking(model_history, test_output):
    train_loss = []
    for i, e in enumerate(model_history["train_loss_list"]):
        train_loss.append({"value": e, "step": i})

    val_loss = []
    for i, e in enumerate(model_history["val_loss_list"]):
        val_loss.append({"value": e, "step": i})

    predictions = []
    for i, e in enumerate(test_output["pred"]):
        predictions.append({"value": e, "step": i})

    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "predictions": predictions,
        "average_test_loss": {"value": test_output["average_test_loss"], "step": 1},
        "accuracy": {"value": test_output["accuracy"], "step": 1},
    }

    return {"metrics": metrics}


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
    mlflow.log_figure(plt, "loss_curve.html")
    return {"loss_curve": plt}


def plot_confusionmatrix(test_output: dict, test_dataloader: DataLoader):
    test_labels = []
    for _, target in test_dataloader:
        test_labels.append(target.item())

    label_predictions = test_output["pred"]

    confusion_matrix = metrics.confusion_matrix(test_labels, label_predictions)
    confusion_matrix = confusion_matrix.transpose()
    labels = [f"{l}" for l in np.unique(test_labels)]
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
    mlflow.log_figure(fig, "confusion_matrix.html")
    return {"confusionmatrix": fig}
