from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    train_model,
    test_model,
    plot_loss,
    plot_confusionmatrix,
    parameter_tracking,
)


def create_pipeline(**kwargs) -> Pipeline:
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
                    "model_tracking": "model_tracking",
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
                outputs=["test_output", "test_tracking"],
                name="test_model",
            ),
            node(plot_loss, inputs="model_history", outputs="loss_curve"),
            node(
                plot_confusionmatrix,
                inputs=["test_output", "test_dataloader"],
                outputs="confusionmatrix",
            ),
            node(
                parameter_tracking,
                inputs=[
                    "params:epochs",
                    "params:learning_rate",
                    "params:loss_func",
                    "params:TRAINING_SIZE",
                    "params:TEST_SIZE",
                    "params:number_of_qubits"
                    "params:two_optimizers"
                ],
                outputs="params_tracking",
                name="parameter_tracking",
            ),
        ],
        inputs={
            "train_dataloader": "train_dataloader",
            "test_dataloader": "test_dataloader",
        },
        outputs={},
        namespace="data_science",
    )
