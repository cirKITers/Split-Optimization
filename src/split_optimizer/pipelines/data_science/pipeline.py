from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
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
                train_model,
                inputs=[
                    "params:epochs",
                    "params:loss_func",
                    "params:learning_rate",
                    "params:two_optimizers",
                    "train_dataloader",
                    "test_dataloader",
                ],
                outputs={
                    "model": "model",
                    "model_history": "model_history",
                },
                name="train_model",
            ),
            node(
                test_model,
                inputs=[
                    "model",
                    "params:loss_func",
                    "params:TEST_SIZE",
                    "test_dataloader",
                ],
                outputs="test_output",
                name="test_model",
            ),
            node(plot_loss, inputs="model_history", outputs="loss_curve"),
            node(
                plot_confusionmatrix,
                inputs=["test_output", "test_dataloader"],
                outputs="confusionmatrix",
            ),
            node(
                mlflow_tracking,
                inputs=["model_history", "test_output"],
                outputs="metrics",
            ),
        ],
        inputs={
            "train_dataloader": "train_dataloader",
            "test_dataloader": "test_dataloader",
        },
        outputs={},
        namespace="data_science",
    )
