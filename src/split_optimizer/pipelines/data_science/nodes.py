from .hybrid_model import Model

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
from .hyperparam_optimizer import Hyperparam_Optimizer

import logging

log = logging.getLogger(__name__)


def train_model_optuna(trial, *args, **kwargs):
    result = train_model(*args, **kwargs)

    # best_val_accuracy = max(result["model_history"]["val_loss_list"])

    return min(result["model_history"]["val_loss_list"])

def append_metrics(metrics, metric, mean=False, prefix=''):
    latest_metric = {}
    for l, m in metric.items():
        if l not in metrics:
            metrics[prefix+l] = []
        if mean:
            latest = np.mean(m)
        else:
            latest = m.item()
        latest_metric[prefix+l] = latest
        metrics[prefix+l].append(latest)
    return metrics, latest_metric

def train_model(
    instructor: Instructor,
) -> Dict:
    train_metrics = {}
    val_metrics = {}
    for epoch in range(instructor.epochs):
        instructor.model.train()
        train_metrics_batch = {"Loss": []}
        for data, target in instructor.train_dataloader:
            _, loss, metrics = instructor.objective_function(data=data, target=target)

            instructor.optimizer.zero_grad()
            loss.backward()
            instructor.optimizer.step(data, target, instructor.objective_function)
            train_metrics_batch["Loss"].append(loss.item())

            train_metrics_batch, _ = append_metrics(train_metrics_batch, metrics)

        train_metrics, train_latest = append_metrics(train_metrics, train_metrics_batch, mean=True, prefix='Train_')

        # log.debug(
        #     f"Training [{100.0*(epoch+1) / instructor.epochs:2.0f}%]\tLoss:{train_metrics['Loss'][-1]:.4f}\tAccuracy:{100.0*train_metrics['Accuracy'][-1]:2.2f}%"
        # )

        metrics_string = [f"\t{l}: {m:3.4f}" for l, m in train_latest.items()]
        log.debug(
            f"Training [{100.0*(epoch+1) / instructor.epochs:2.0f}%]{str().join(metrics_string)}"
        )

        instructor.model.eval()
        with torch.no_grad():
            val_metrics_batch = {"Loss": []}
            for data, target in instructor.test_dataloader:
                _, loss, metrics = instructor.objective_function(data=data, target=target)

                val_metrics_batch["Loss"].append(loss.item())
                val_metrics_batch, _ = append_metrics(val_metrics_batch, metrics)

        val_metrics, val_latest = append_metrics(val_metrics, val_metrics_batch, mean=True, prefix='Val_')

        metrics_string = [f"\t{l}: {m:3.4f}" for l, m in val_latest.items()]
        log.debug(
            f"Training [{100.0*(epoch+1) / instructor.epochs:2.0f}%]{str().join(metrics_string)}"
        )

        mlflow.log_metrics(train_latest | val_latest, step=epoch)


    model_history = {
        "train_loss_list": train_metrics["Train_Loss"],
        "val_loss_list": val_metrics["Val_Loss"],
    }

    return {
        "model": instructor.model,
        "model_history": model_history,
    }


def test_model(
    instructor: Instructor,
    model: Model
) -> Dict:
    instructor.model = model
    instructor.model.eval()
    with torch.no_grad():
        test_metrics_batch = {"Loss": [], "Accuracy": []}
        predictions = []
        for data, target in instructor.test_dataloader:
            pred, loss, metrics = instructor.objective_function(data=data, target=target)

            test_metrics_batch["Loss"].append(loss.item())
            test_metrics_batch["Accuracy"].append(metrics["Accuracy"].item())
            predictions += pred.tolist()

        label_predictions = []
        for i in predictions:
            label_predictions.append(np.argmax(i).item())

    test_output = {
        "average_test_loss": np.mean(test_metrics_batch["Loss"]),
        "accuracy": np.mean(test_metrics_batch["Accuracy"]),
        "pred": label_predictions,
    }

    return {"test_output": test_output}


def create_instructor(
    model: nn.Module,
    loss_func: str,
    optimizer: Dict,
    epochs: int,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    class_weights_train: List,
    torch_seed: int,
):
    instructor = Instructor(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        epochs=epochs,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        class_weights_train=class_weights_train,
        torch_seed=torch_seed,
    )

    return {"instructor": instructor}


def create_model(n_qubits: int, n_layers: int, classes: List, data_reupload: int, quant_status:int):
    model = Model(
        n_qubits=n_qubits,
        n_layers=n_layers,
        classes=classes,
        data_reupload=data_reupload,
        quant_status=quant_status,
    )

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
        if len(target) > 1:
            test_labels += [t for t in target.tolist()]
        else:
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


def create_hyperparam_optimizer(
    n_trials: str,
    timeout: int,
    optuna_path: str,
    optuna_sampler_seed: int,
    pool_process: bool,
    pruner_startup_trials: int,
    pruner_warmup_steps: int,
    pruner_interval_steps: int,
    pruner_min_trials: int,
    selective_optimization: bool,
    resume_study: bool,
    n_jobs: int,
    run_id: str,
    n_qubits_range_quant: int,
    n_layers_range_quant: int,
    classes: List,
    data_reupload_range_quant: List,
    quant_status: int,
    loss_func: str,
    optimizer_range: Dict,
    epochs: List,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    class_weights_train: List,
    torch_seed: int,
) -> Hyperparam_Optimizer:
    if run_id is None:
        name = mlflow.active_run().info.run_id
    else:
        name = run_id

    hyperparam_optimizer = Hyperparam_Optimizer(
        name=name,
        seed=optuna_sampler_seed,
        n_trials=n_trials,
        timeout=timeout,
        path=optuna_path,
        n_jobs=n_jobs,
        selective_optimization=selective_optimization,
        resume_study=resume_study,
        pool_process=pool_process,
        pruner_startup_trials=pruner_startup_trials,
        pruner_warmup_steps=pruner_warmup_steps,
        pruner_interval_steps=pruner_interval_steps,
        pruner_min_trials=pruner_min_trials,
    )

    hyperparam_optimizer.set_variable_parameters(
        {
            "n_qubits_range_quant": n_qubits_range_quant,
            "n_layers_range_quant": n_layers_range_quant,
            "data_reupload_range_quant": data_reupload_range_quant,
        },
        {
            "optimizer_range": optimizer_range,
        },
    )

    hyperparam_optimizer.set_fixed_parameters(
        {"classes": classes, "quant_status":quant_status},
        {
            "model": None,  # this must be overwritten later in the optimization step and just indicates the difference in implementation here
            "loss_func": loss_func,
            "epochs": epochs,
            "train_dataloader": train_dataloader,
            "test_dataloader": test_dataloader,
            "class_weights_train": class_weights_train,
            "torch_seed": torch_seed,
        },
    )

    hyperparam_optimizer.create_model = Model
    hyperparam_optimizer.create_instructor = Instructor
    hyperparam_optimizer.objective = train_model_optuna

    return {"hyperparam_optimizer": hyperparam_optimizer}


def run_optuna(hyperparam_optimizer: Hyperparam_Optimizer):
    hyperparam_optimizer.minimize()

    # artifacts = hyperparam_optimizer.log_study()

    return {}
