from .hybrid_model import Net

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import metrics
from typing import Dict, List
import plotly.express as px
from .optimizer import initialize_optimizer
import mlflow

# epochs: int, TRAINING_SIZE: int, dataset: list[np.ndarray]
from torch.utils.data.dataloader import DataLoader
import plotly.graph_objects as go

class Instructor():
    def __init__(self,
        model: nn.Module,
        loss_func: str,
        learning_rate: float,
        optimizer_list: List,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        class_weights_train: List,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        if loss_func == "CrossEntropyLoss":
            self.train_loss = nn.CrossEntropyLoss(weight=class_weights_train)
            self.test_loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f"{loss_func} is not a loss function in [CrossEntropyLoss]"
            )  # TODO: shall we actually add more loss functions?

        self.optimizer = initialize_optimizer(model, learning_rate, optimizer_list)

    def objective_function(self, data, target, train=True):
        output = self.model(data)
        loss = self.train_loss(output, target)
        return loss

        
def train_model(
    instructor:Instructor,
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
                loss = instructor.objective_function(data=data, target=target, train=False)

                val_loss.append(loss.item())

        val_loss_list.append(np.mean(val_loss))

    model_history = {"train_loss_list": train_loss_list, "val_loss_list": val_loss_list}

    return {
        "model": instructor.model,
        "model_history": model_history,
    }


def test_model(
    instructor:Instructor,
) -> Dict:
    instructor.model.eval()

    with torch.no_grad():
        correct = 0
        test_loss_list = []
        predictions = []
        for data, target in instructor.test_dataloader:
            output = instructor.model(data)

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

    return {
        "instructor":instructor
    }

def create_model(
        n_qubits:int,
        classes:List
):
    model = Net(n_qubits=n_qubits,
    classes=classes)

    return {
        "model":model
    }



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


def train_optuna(training_node: callable, instructor: Instructor, trial, start_epoch=1, enabled_modes=["train", "val"]):

    result = training_node(trial,
        start_epoch=start_epoch, enabled_modes=enabled_modes
    )  # returns a dict of e.g. the model, checkpoints and the gradients

    return {
        'metrics': result['metrics']
    }

def create_hyperparam_optimizer(
    n_classes,
    n_momenta,
    model_sel,
    n_blocks_range: List,
    dim_feedforward_range: List,
    n_layers_mlp_range: List,
    n_additional_mlp_layers_range: List,
    n_final_mlp_layers_range: List,
    skip_block: bool,
    skip_global: bool,
    dropout_rate_range: List,
    batchnorm: bool,
    symmetrize: bool,
    data_reupload_range_quant: bool,
    n_layers_vqc_range_quant: List,
    padding_dropout: bool,
    predefined_vqc_range_quant: List,
    predefined_iec: str,
    measurement: str,
    backend: str,
    n_shots_range_quant: int,
    n_fsps: int,

    device: str,
    initialization_constant: float,
    initialization_offset: float,
    parameter_seed:int, 
    dataset_lca_and_leaves: Dict,
    learning_rate_range: List,
    learning_rate_decay_range: List,
    decay_after: float,
    batch_size_range: List,
    epochs: int,
    normalize: str,
    normalize_individually: bool,
    zero_mean: bool,
    plot_mode: str,
    plotting_rows: int,
    log_gradients: bool,
    gradients_clamp: int,
    gradients_spreader: float,
    torch_seed: int,
    gradient_curvature_threshold_range_quant: float,
    gradient_curvature_history_range_quant: int,
    quantum_optimizer_range_quant: List,
    quantum_momentum: float,
    quantum_learning_rate_range_quant: List,
    quantum_learning_rate_decay_range_quant: List,
    classical_optimizer: str,
    detectAnomaly: bool,
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
) -> Hyperparam_Optimizer:

    if "q" in model_sel:
        toggle_classical_quant = True
    else:
        toggle_classical_quant = False

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
        toggle_classical_quant=toggle_classical_quant,
        resume_study=resume_study,
        pool_process=pool_process,
        pruner_startup_trials=pruner_startup_trials,
        pruner_warmup_steps=pruner_warmup_steps,
        pruner_interval_steps=pruner_interval_steps,
        pruner_min_trials=pruner_min_trials
    )

    hyperparam_optimizer.set_variable_parameters(
        {
            "n_blocks_range": n_blocks_range,
            "dim_feedforward_range": dim_feedforward_range,
            "n_layers_mlp_range": n_layers_mlp_range,
            "n_additional_mlp_layers_range": n_additional_mlp_layers_range,
            "n_final_mlp_layers_range": n_final_mlp_layers_range,
            "dropout_rate_range": dropout_rate_range,
            "data_reupload_range_quant": data_reupload_range_quant,
            "n_layers_vqc_range_quant": n_layers_vqc_range_quant,
            "predefined_vqc_range_quant": predefined_vqc_range_quant,
            # "initialization_constant_range_quant": initialization_constant_range_quant,
            # "initialization_offset_range_quant": initialization_offset_range_quant,
            "n_shots_range_quant": n_shots_range_quant,
        },
        {
            "learning_rate_range": learning_rate_range,
            "learning_rate_decay_range": learning_rate_decay_range,
            "quantum_learning_rate_range_quant": quantum_learning_rate_range_quant,
            "quantum_learning_rate_decay_range_quant": quantum_learning_rate_decay_range_quant,
            "batch_size_range": batch_size_range,
            "gradient_curvature_history_range_quant": gradient_curvature_history_range_quant,
            "quantum_optimizer_range_quant": quantum_optimizer_range_quant,
            "gradient_curvature_threshold_range_quant": gradient_curvature_threshold_range_quant,
        },
    )

    hyperparam_optimizer.set_fixed_parameters(
        {
            "n_classes": n_classes,
            "n_momenta": n_momenta,
            "model_sel": model_sel,
            "skip_block":skip_block,
            "skip_global":skip_global,
            "batchnorm": batchnorm,
            "symmetrize": symmetrize,
            "padding_dropout": padding_dropout,
            "predefined_iec": predefined_iec,
            "measurement": measurement,
            "backend": backend,
            "n_fsps": n_fsps,
            "device": device,
            "initialization_constant": initialization_constant,
            "initialization_offset": initialization_offset,
            "parameter_seed": parameter_seed,
        },
        {
            "model": None,  # this must be overwritten later in the optimization step and just indicates the difference in implementation here
            "dataset_lca_and_leaves": dataset_lca_and_leaves,
            "n_classes": n_classes,
            "epochs": epochs,
            "normalize": normalize,
            "normalize_individually": normalize_individually,
            "zero_mean": zero_mean,
            "plot_mode": plot_mode,
            "plotting_rows": plotting_rows,
            "detectAnomaly": detectAnomaly,
            "log_gradients": log_gradients,
            "device": device,
            "n_fsps": n_fsps,
            "decay_after": decay_after,
            "gradients_clamp": gradients_clamp,
            "gradients_spreader": gradients_spreader,
            "torch_seed": torch_seed,
            "quantum_momentum": quantum_momentum,
            "classical_optimizer": classical_optimizer,
            "logging": False
        },
    )

    hyperparam_optimizer.create_model = create_model
    hyperparam_optimizer.create_instructor = create_instructor
    hyperparam_optimizer.objective = train_optuna

    return {"hyperparam_optimizer": hyperparam_optimizer}


def run_optuna(hyperparam_optimizer: Hyperparam_Optimizer):

    hyperparam_optimizer.minimize()

    # artifacts = hyperparam_optimizer.log_study()

    return {}


