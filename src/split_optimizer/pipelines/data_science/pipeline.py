from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    create_model,
    create_instructor,
    train_model,
    test_model,
    plot_loss,
    plot_confusionmatrix,
    mlflow_tracking,
)


def create_training_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                create_model,
                inputs={
                    "n_qubits": "params:n_qubits",
                    "n_layers": "params:n_layers",
                    "classes": "params:classes",
                },
                outputs={
                    "model": "model",
                },
                name="create_model",
            ),
            node(
                create_instructor,
                inputs={
                    "model": "model",
                    "loss_func": "params:loss_func",
                    "optimizer": "params:optimizer",
                    "train_dataloader": "train_dataloader",
                    "test_dataloader": "test_dataloader",
                    "class_weights_train": "class_weights_train",
                },
                outputs={
                    "instructor": "instructor",
                },
                name="create_instructor",
            ),
            node(
                train_model,
                inputs={
                    "instructor": "instructor",
                    "epochs": "params:epochs",
                },
                outputs={
                    "model": "trained_model",
                    "model_history": "model_history",
                },
                name="train_model",
            ),
            node(
                test_model,
                inputs={
                    "instructor": "instructor",
                    "model": "trained_model",
                },
                outputs={"test_output": "test_output"},
                name="test_model",
            ),
            node(
                plot_loss, inputs="model_history", outputs={"loss_curve": "loss_curve"}
            ),
            node(
                plot_confusionmatrix,
                inputs=["test_output", "test_dataloader"],
                outputs={"confusionmatrix": "confusionmatrix"},
            ),
            node(
                mlflow_tracking,
                inputs=["model_history", "test_output"],
                outputs={"metrics": "metrics"},
            ),
        ],
        inputs={
            "train_dataloader": "train_dataloader",
            "test_dataloader": "test_dataloader",
            "class_weights_train": "class_weights_train",
        },
        outputs={},
        namespace="data_science",
    )
