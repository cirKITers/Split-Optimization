import numpy as np
from kedro.framework.session import KedroSession
import json
from pathlib import Path
from kedro.framework.startup import bootstrap_project

def run_training():
    with KedroSession.create() as session:
        output = session.run(pipeline_name="test_pipeline")
        data_catalog = session.load_context().catalog

        data_catalog = session.load_context().config_loader["catalog"]
        metrics_fig = data_catalog["data_science.metrics_fig"]["data_set"]
        filepath = metrics_fig["filepath"]

        with open(filepath, "r") as file:
            metrics = json.load(file)

        train_loss = metrics["data"][0]["y"]
        train_accuracy = metrics["data"][1]["y"]
        val_accuracy = metrics["data"][3]["y"]
        parameters = session.load_context().config_loader["parameters"][
            "data_processing"
        ]

    return train_loss, train_accuracy, val_accuracy, parameters


class TestTraining:
    bootstrap_project(Path.cwd())
    train_loss, train_accuracy, val_accuracy, parameters = run_training()
    second_train_loss, second_train_accuracy, second_val_accuracy, _ = run_training()

    def test_training(self):
        coincidence_accuracy = 1 / len(self.parameters["classes"])
        # check if accuracy is better than the minimum coincidence case
        assert (
            self.train_accuracy[-1] > coincidence_accuracy
        ), f"train accuracy should be higher than {coincidence_accuracy}"
        assert (
            self.val_accuracy[-1] > coincidence_accuracy
        ), f"validation accuracy should be higher than {coincidence_accuracy} "

    def test_reproducability(self):
        assert np.array_equal(
            self.second_train_loss, self.train_loss
        ), "training is not consistent"

def test_optimizer():
    bootstrap_project(Path.cwd())
    # iterate all optimizer, run a training
    for i in ["SGD", "Adam"]:
        for p in ["Adam", "SPSA", "SGD", "NGD", "QNG"]:
            params = {
                "data_science": {
                    "optimizer": {
                        "split": {"classical": {"name": i}, "quantum": {"name": p}}
                    }
                }
            }
            # create kedroSession and change optimizer by passing extra_params
            with KedroSession.create(extra_params=params) as session:
                parameters = session.load_context().params["data_science"]
                optimizer = parameters["optimizer"]
                if "split" not in optimizer:
                    raise ValueError("Enable Split Optimizer in config")
                output = session.run(pipeline_name="debug_pipeline")
