from .nodes import prepare_data
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                prepare_data,
                inputs=["params:train_filepath", "params:test_filepath", "params:batch_size", "params:TRAINING_SIZE", "params:TEST_SIZE", "params:seed"], 
                outputs={
                    "train_dataloader": "train_dataloader",
                    "test_dataloader": "test_dataloader",
                },
                name="train_model",
            )
        ],
        inputs={},
        outputs={
                    "train_dataloader": "train_dataloader",
                    "test_dataloader": "test_dataloader",
                },
        namespace="data_processing",
    )
