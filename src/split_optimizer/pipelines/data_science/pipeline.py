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
    create_hyperparam_optimizer,
    run_optuna,
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
                    "data_reupload": "params:data_reupload",
                    "quant_status": "params:quant_status",
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
                    "epochs": "params:epochs",
                    "train_dataloader": "train_dataloader",
                    "test_dataloader": "test_dataloader",
                    "class_weights_train": "class_weights_train",
                    "torch_seed": "params:torch_seed",
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
                },
                outputs={
                    "model": "trained_model",
                    "metrics": "metrics",
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
                plot_loss,
                inputs={
                    "epochs": "params:epochs",
                    "metrics": "metrics",
                },
                outputs={"metrics_fig": "metrics_fig"},
                name="plot_loss",
            ),
            node(
                plot_confusionmatrix,
                inputs=["test_output", "test_dataloader"],
                outputs={"confusionmatrix": "confusionmatrix"},
                name="plot_confusionmatrix",
            ),
            # node(
            #     mlflow_tracking,
            #     inputs=["model_history", "test_output"],
            #     outputs={"metrics": "metrics"},
            # ),
        ],
        inputs={
            "train_dataloader": "train_dataloader",
            "test_dataloader": "test_dataloader",
            "class_weights_train": "class_weights_train",
        },
        outputs={},
        namespace="data_science",
    )


def create_hyperparam_opt_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                create_hyperparam_optimizer,
                inputs={
                    "optuna_n_trials": "params:optuna_n_trials",
                    "optuna_timeout": "params:optuna_timeout",
                    "optuna_enabled_hyperparameters": "params:optuna_enabled_hyperparameters",
                    "optuna_optimization_metric": "params:optuna_optimization_metric",
                    "optuna_path": "params:optuna_path",
                    "optuna_sampler": "params:optuna_sampler",
                    "optuna_sampler_seed": "params:optuna_sampler_seed",
                    "optuna_pool_process": "params:optuna_pool_process",
                    "pruner": "params:pruner",
                    "pruner_startup_trials": "params:pruner_startup_trials",
                    "pruner_warmup_steps": "params:pruner_warmup_steps",
                    "pruner_interval_steps": "params:pruner_interval_steps",
                    "pruner_min_trials": "params:pruner_min_trials",
                    "optuna_selective_optimization": "params:optuna_selective_optimization",
                    "optuna_resume_study": "params:optuna_resume_study",
                    "optuna_n_jobs": "params:optuna_n_jobs",
                    "optuna_run_id": "params:optuna_run_id",
                    "n_qubits": "params:n_qubits",
                    "n_qubits_range_quant": "params:n_qubits_range_quant",
                    "n_layers": "params:n_layers",
                    "n_layers_range_quant": "params:n_layers_range_quant",
                    "classes": "params:classes",
                    "data_reupload": "params:data_reupload",
                    "data_reupload_range_quant": "params:data_reupload_range_quant",
                    "quant_status": "params:quant_status",
                    "loss_func": "params:loss_func",
                    "epochs": "params:epochs",
                    "optimizer": "params:optimizer",
                    "optimizer_choice": "params:optimizer_choice",
                    "train_dataloader": "train_dataloader",
                    "test_dataloader": "test_dataloader",
                    "class_weights_train": "class_weights_train",
                    "torch_seed": "params:torch_seed",
                },
                outputs={
                    "hyperparam_optimizer": "hyperparam_optimizer",
                },
                name="create_hyperparam_optimizer",
            ),
            node(
                run_optuna,
                inputs={
                    "hyperparam_optimizer": "hyperparam_optimizer",
                    "optuna_selected_parallel_params": "params:optuna_selected_parallel_params",
                    "optuna_selected_slice_params": "params:optuna_selected_slice_params",
                },
                outputs={},
                name="run_optuna",
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
