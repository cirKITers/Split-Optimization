from pathlib import Path
import numpy as np
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
import json


""" def test_training():
    bootstrap_project(Path.cwd())
    with KedroSession.create() as preparation_session:
        data_preparation_output = preparation_session.run(pipeline_name="preprocessing")
        output = preparation_session.run(pipeline_name="training")
 """
class TestTraining:
    
    bootstrap_project(Path.cwd())
    



    def test_training(self):



        with KedroSession.create() as preparation_session:
            data_preparation_output = preparation_session.run(pipeline_name="preprocessing")
        
        with KedroSession.create() as preparation_session:
            output = preparation_session.run(pipeline_name="training")

        data_catalog = session.load_context().config_loader["catalog"]
        metrics_fig = data_catalog["data_science.metrics_fig"]["data_set"]
        filepath = metrics_fig["filepath"]

        with open(filepath, "r") as file:
            metrics = json.load(file)

        train_loss = metrics["data"][0]["y"]

        # start = train_loss[0]
        # for i, p in enumerate(train_loss):
        #    assert p < start / (0.5 * (i + 1)), "loss does not decrease efficiently"

        train_accuracy = metrics["data"][1]["y"]
        val_accuracy = metrics["data"][3]["y"]

        parameters = session.load_context().config_loader["parameters"]["data_processing"]
        coincidence_accuracy = 1 / len(parameters["classes"])

        # check if accuracy is better than the minimum coincidence case
        assert (
            train_accuracy[-1] > coincidence_accuracy
        ), f"train accuracy should be higher than {coincidence_accuracy}"
        assert (
            val_accuracy[-1] > coincidence_accuracy
        ), f"validation accuracy should be higher than {coincidence_accuracy} "

        # check reproducability
        session.close()
        bootstrap_project(Path.cwd())
        with KedroSession.create() as session:
            output = session.run(pipeline_name="__default__")

        data_catalog = session.load_context().config_loader["catalog"]
        metrics_fig = data_catalog["data_science.metrics_fig"]["data_set"]
        filepath = metrics_fig["filepath"]

        with open(filepath, "r") as file:
            metrics = json.load(file)

        second_train_loss = metrics["data"][0]["y"]

        assert np.array_equal(second_train_loss, train_loss), "training is not consistent"


    def test_optimizer(self):
        # iterate all optimizer, run a training 
        bootstrap_project(Path.cwd())
        
        for i in ["SGD", "Adam"]:
            for p in ["Adam", "SPSA", "SGD", "NGD", "QNG"]:
                session = KedroSession.create()
                parameters = session.load_context().config_loader["parameters"]["data_science"]
                optimizer = parameters["optimizer"]
                parameters["epochs"] = 2
                optimizer["split"]["classical"]["name"] = i
                optimizer["split"]["quantum"]["name"] = p
                output = session.run(pipeline_name="debug_pipeline")
                session.close()
