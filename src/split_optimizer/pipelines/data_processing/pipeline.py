from .nodes import (
    load_data,
    select_classes,
    reduce_size,
    onehot,
    normalize,
    create_dataloader,
    calculate_class_weights,
)
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                load_data,
                inputs=[],
                outputs={
                    "train_dataset_full": "train_dataset_full",
                    "test_dataset_full": "test_dataset_full",
                },
                name="load_data",
            ),
            node(
                select_classes,
                inputs={
                    "train_dataset_full": "train_dataset_full",
                    "test_dataset_full": "test_dataset_full",
                    "classes": "params:classes",
                },
                outputs={
                    "train_dataset_selected": "train_dataset_selected",
                    "test_dataset_selected": "test_dataset_selected",
                },
                name="select_classes",
            ),
            node(
                reduce_size,
                inputs={
                    "train_dataset_selected": "train_dataset_selected",
                    "test_dataset_selected": "test_dataset_selected",
                    "TRAINING_SIZE": "params:TRAINING_SIZE",
                    "TEST_SIZE": "params:TEST_SIZE",
                },
                outputs={
                    "train_dataset_reduced": "train_dataset_reduced",
                    "test_dataset_reduced": "test_dataset_reduced",
                },
                name="reduce_size",
            ),
            node(
                onehot,
                inputs={
                    "train_dataset_reduced": "train_dataset_reduced",
                    "test_dataset_reduced": "test_dataset_reduced",
                    "TRAINING_SIZE": "params:TRAINING_SIZE",
                    "TEST_SIZE": "params:TEST_SIZE",
                    "classes": "params:classes",
                },
                outputs={
                    "test_dataset_onehot": "test_dataset_onehot",
                    "train_dataset_onehot": "train_dataset_onehot",
                },
                name="onehot",
            ),
            node(
                normalize,
                inputs={
                    "test_dataset_onehot": "test_dataset_onehot",
                    "train_dataset_onehot": "train_dataset_onehot",
                },
                outputs={
                    "test_dataset": "test_dataset",
                    "train_dataset": "train_dataset",
                },
                name="normalize",
            ),
            node(
                create_dataloader,
                inputs={
                    "train_dataset": "train_dataset",
                    "test_dataset": "test_dataset",
                    "batch_size": "params:batch_size",
                    "seed": "params:seed",
                },
                outputs={
                    "train_dataloader": "train_dataloader",
                    "test_dataloader": "test_dataloader",
                },
                name="create_dataloader",
            ),
            node(
                calculate_class_weights,
                inputs={
                    "train_dataset_reduced": "train_dataset_reduced",
                    "classes": "params:classes",
                    "TRAINING_SIZE": "params:TRAINING_SIZE",
                },
                outputs={
                    "class_weights_train": "class_weights_train",
                },
                name="calculate_class_weights",
            ),
        ],
        inputs={},
        outputs={
            "train_dataloader": "train_dataloader",
            "test_dataloader": "test_dataloader",
            "class_weights_train": "class_weights_train",
        },
        namespace="data_processing",
    )
