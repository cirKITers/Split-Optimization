from .nodes import load_data, format_data, create_dataloader
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                load_data,
                inputs=[],
                outputs={
                    "x_train_full": "x_train_full",
                    "y_train_full": "y_train_full",
                    "x_test_full": "x_test_full",
                    "y_test_full": "y_test_full",
                },
                name="load_data",
            ),
            node(
                format_data,
                inputs=[
                    "x_train_full",
                    "y_train_full",
                    "x_test_full",
                    "y_test_full",
                    "params:TRAINING_SIZE",
                    "params:TEST_SIZE",
                    "params:number_classes",
                ],
                outputs={
                    "x_train": "x_train",
                    "y_train": "y_train",
                    "x_test": "x_test",
                    "y_test": "y_test",
                },
                name="format_data",
            ),
            node(
                create_dataloader,
                inputs=[
                    "x_train",
                    "y_train",
                    "x_test",
                    "y_test",
                    "params:batch_size",
                    "params:seed",
                ],
                outputs={
                    "train_dataloader": "train_dataloader",
                    "test_dataloader": "test_dataloader",
                },
                name="create_dataloader",
            ),
        ],
        inputs={},
        outputs={
            "train_dataloader": "train_dataloader",
            "test_dataloader": "test_dataloader",
        },
        namespace="data_processing",
    )
