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
from .metrics import metrics
from .design import design

import logging

log = logging.getLogger(__name__)


def train_model_optuna(trial, *args, **kwargs):
    result = train_model(*args, **kwargs)

    min_metric = {}
    for l, m in result["metrics"].items():
        name = l.replace("Train_", "").replace("Val_", "")
        if metrics[name]["s"] < 0:
            min_metric[l] = max(m)
        else:
            min_metric[l] = min(m)

    return min_metric


def append_metrics(metrics, metric, mean=False, prefix=""):
    latest_metric = {}
    for l, m in metric.items():
        act_key = prefix + l
        if act_key not in metrics.keys():
            metrics[act_key] = []
        if mean:
            latest = np.mean(m)
        else:
            latest = m.item()
        latest_metric[act_key] = latest
        metrics[act_key].append(latest)
    return metrics, latest_metric


def train_model(instructor: Instructor) -> Dict:
    train_metrics = {}
    val_metrics = {}
    for epoch in range(instructor.epochs):
        instructor.model.train()
        train_metrics_batch = {}
        for data, target in instructor.train_dataloader:
            _, loss, metrics = instructor.objective_function(data=data, target=target)

            instructor.optimizer.zero_grad()
            loss.backward()
            instructor.optimizer.step(
                data,
                target,
                lambda **kwargs: instructor.objective_function(**kwargs)[1],
            )

            train_metrics_batch, _ = append_metrics(train_metrics_batch, metrics)

        train_metrics, train_latest = append_metrics(
            train_metrics, train_metrics_batch, mean=True, prefix="Train_"
        )

        metrics_string = [f"\t{l}: {m:3.4f}" for l, m in train_latest.items()]
        log.debug(
            f"Training [{100.0*(epoch+1) / instructor.epochs:2.0f}%]{str().join(metrics_string)}"
        )

        instructor.model.eval()
        with torch.no_grad():
            val_metrics_batch = {}
            for data, target in instructor.test_dataloader:
                _, loss, metrics = instructor.objective_function(
                    data=data, target=target
                )

                val_metrics_batch, _ = append_metrics(val_metrics_batch, metrics)

        val_metrics, val_latest = append_metrics(
            val_metrics, val_metrics_batch, mean=True, prefix="Val_"
        )

        metrics_string = [f"\t{l}: {m:3.4f}" for l, m in val_latest.items()]
        log.debug(
            f"Training [{100.0*(epoch+1) / instructor.epochs:2.0f}%]{str().join(metrics_string)}"
        )

        mlflow.log_metrics(train_latest | val_latest, step=epoch)

        instructor.report_callback(metrics=train_latest | val_latest, step=epoch)
        if instructor.early_stop_callback():
            log.info(f"Early stopping triggered in epoch {epoch}. Stopping training.")
            break

    return {
        "model": instructor.model,
        "metrics": train_metrics | val_metrics,
    }


def test_model(instructor: Instructor, model: Model) -> Dict:
    instructor.model = model
    instructor.model.eval()
    with torch.no_grad():
        test_metrics_batch = {}
        predictions = []
        for data, target in instructor.test_dataloader:
            pred, _, metrics = instructor.objective_function(data=data, target=target)

            test_metrics_batch, _ = append_metrics(test_metrics_batch, metrics)
            predictions += pred.tolist()

        label_predictions = []
        for i in predictions:
            label_predictions.append(np.argmax(i).item())

    test_output = {
        "average_test_loss": np.mean(test_metrics_batch["CrossEntropy"]),
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


def create_model(
    n_qubits: int,
    n_layers: int,
    classes: List,
    data_reupload: int,
    quant_status: int,
    n_shots: int,
):
    model = Model(
        n_qubits=n_qubits,
        n_layers=n_layers,
        classes=classes,
        data_reupload=data_reupload,
        quant_status=quant_status,
        n_shots=n_shots,
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


def plot_loss(epochs: int, metrics: dict) -> plt.figure:
    epochs = list(range(1, epochs + 1))

    plt = go.Figure(
        [
            go.Scatter(
                x=epochs,
                y=m,
                mode=design.scatter_markers,
                name=f"Training {l}" if "Train_" in l else f"Validation {l}",
            )
            for l, m in metrics.items()
        ]
    )
    plt.update_layout(
        yaxis=dict(
            title="Metric",
            showgrid=design.showgrid,
        ),
        xaxis=dict(
            title="Epochs",
            showgrid=design.showgrid,
        ),
        title=dict(
            text=f"Training and Validation Metrics"
            if design.print_figure_title
            else "",
            font=dict(
                size=design.title_font_size,
            ),
        ),
        hovermode="x",
        font=dict(
            size=design.legend_font_size,
        ),
        template="simple_white",
    )
    mlflow.log_figure(plt, "metrics_fig.html")
    return {"metrics_fig": plt}


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
    optuna_n_trials: str,
    optuna_timeout: int,
    optuna_enabled_hyperparameters: List,
    optuna_optimization_metric: List,
    optuna_path: str,
    optuna_sampler: str,
    optuna_sampler_seed: int,
    optuna_pool_process: bool,
    pruner: str,
    pruner_startup_trials: int,
    pruner_warmup_steps: int,
    pruner_interval_steps: int,
    pruner_min_trials: int,
    optuna_selective_optimization: bool,
    optuna_resume_study: bool,
    optuna_n_jobs: int,
    optuna_run_id: str,
    n_qubits: int,
    n_qubits_range_quant: int,
    n_layers: int,
    n_layers_range_quant: int,
    classes: List,
    data_reupload: List,
    data_reupload_range_quant: List,
    quant_status: int,
    n_shots: int,
    loss_func: str,
    optimizer: Dict,
    optimizer_choice: Dict,
    epochs: List,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    class_weights_train: List,
    torch_seed: int,
) -> Hyperparam_Optimizer:
    if optuna_run_id is None:
        name = mlflow.active_run().info.run_id
    else:
        name = optuna_run_id

    hyperparam_optimizer = Hyperparam_Optimizer(
        name=name,
        sampler=optuna_sampler,
        seed=optuna_sampler_seed,
        n_trials=optuna_n_trials,
        timeout=optuna_timeout,
        enabled_hyperparameters=optuna_enabled_hyperparameters,
        optimization_metric=optuna_optimization_metric,
        path=optuna_path,
        n_jobs=optuna_n_jobs,
        selective_optimization=optuna_selective_optimization,
        resume_study=optuna_resume_study,
        pool_process=optuna_pool_process,
        pruner=pruner,
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
            "optimizer_choice": optimizer_choice,
        },
    )

    hyperparam_optimizer.set_fixed_parameters(
        {
            "classes": classes,
            "quant_status": quant_status,
            "n_shots": n_shots,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "data_reupload": data_reupload,
        },
        {
            "model": None,  # this must be overwritten later in the optimization step and just indicates the difference in implementation here
            "loss_func": loss_func,
            "epochs": epochs,
            "train_dataloader": train_dataloader,
            "test_dataloader": test_dataloader,
            "class_weights_train": class_weights_train,
            "torch_seed": torch_seed,
            "optimizer": optimizer,
        },
    )

    hyperparam_optimizer.create_model = Model
    hyperparam_optimizer.create_instructor = Instructor
    hyperparam_optimizer.objective = train_model_optuna

    return {"hyperparam_optimizer": hyperparam_optimizer}


def run_optuna(
    hyperparam_optimizer: Hyperparam_Optimizer,
    optuna_selected_parallel_params,
    optuna_selected_slice_params,
):
    hyperparam_optimizer.minimize()

    try:
        hyperparam_optimizer.log_study(
            selected_parallel_params=optuna_selected_parallel_params,
            selected_slice_params=optuna_selected_slice_params,
        )
    except Exception as e:
        log.exception("Error while logging study")

    return {}
