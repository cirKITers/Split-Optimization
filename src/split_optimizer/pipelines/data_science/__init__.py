"""
This is a boilerplate pipeline 'mnist_processing'
generated using Kedro 0.18.1
"""

from .pipeline import create_training_pipeline, create_hyperparam_opt_pipeline

__all__ = ["create_training_pipeline", "create_hyperparam_opt_pipeline"]

__version__ = "0.1"
