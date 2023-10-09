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
                    "train_metrics": "train_metrics",
                    "val_metrics": "val_metrics",
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
                    "train_metrics": "train_metrics",
                    "val_metrics": "val_metrics",
                },
                outputs={"loss_curve": "loss_curve"},
            ),
            node(
                plot_confusionmatrix,
                inputs=["test_output", "test_dataloader"],
                outputs={"confusionmatrix": "confusionmatrix"},
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
                    "n_trials": "params:n_trials",
                    "timeout": "params:timeout",
                    "optuna_path": "params:optuna_path",
                    "optuna_sampler_seed": "params:optuna_sampler_seed",
                    "pool_process": "params:pool_process",
                    "pruner_startup_trials": "params:pruner_startup_trials",
                    "pruner_warmup_steps": "params:pruner_warmup_steps",
                    "pruner_interval_steps": "params:pruner_interval_steps",
                    "pruner_min_trials": "params:pruner_min_trials",
                    "selective_optimization": "params:selective_optimization",
                    "resume_study": "params:resume_study",
                    "n_jobs": "params:n_jobs",
                    "run_id": "params:run_id",
                    "n_qubits_range_quant": "params:n_qubits_range_quant",
                    "n_layers_range_quant": "params:n_layers_range_quant",
                    "classes": "params:classes",
                    "data_reupload_range_quant": "params:data_reupload_range_quant",
                    "quant_status": "params:quant_status",
                    "loss_func": "params:loss_func",
                    "epochs": "params:epochs",
                    "optimizer_range": "params:optimizer_range",
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
