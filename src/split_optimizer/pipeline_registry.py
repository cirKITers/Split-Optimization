"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


from split_optimizer.pipelines import data_processing, data_science


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = data_processing.create_pipeline()
    data_science_training_pipeline = data_science.create_training_pipeline()
    post_processing_pipeline = data_science.create_postprocessing_pipeline()
    data_science_hyperparam_opt_pipeline = data_science.create_hyperparam_opt_pipeline()

    default_pipeline = data_processing_pipeline + data_science_training_pipeline

    return {
        "__default__": data_processing_pipeline + data_science_training_pipeline + post_processing_pipeline,
        "debug_pipeline": data_processing_pipeline + data_science_training_pipeline + post_processing_pipeline,
        "test_pipeline": data_processing_pipeline + data_science_training_pipeline,
        "optuna_pipeline": data_processing_pipeline
        + data_science_hyperparam_opt_pipeline,
        "preprocessing": data_processing_pipeline,
        "training": data_science_training_pipeline,
        "hyperparameter_opt": data_science_hyperparam_opt_pipeline,
    }
