from pathlib import Path
import numpy as np
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
import torch


def run_preprocessing():
    bootstrap_project(Path.cwd())
    with KedroSession.create() as session:
        output = session.run(pipeline_name="preprocessing")

    parameters = session.load_context().config_loader["parameters"]["data_processing"]
    train_dataloader = output["train_dataloader"]
    test_dataloader = output["test_dataloader"]

    return parameters, train_dataloader, test_dataloader

def second_run_preprocessing():
    bootstrap_project(Path.cwd())
    with KedroSession.create() as second_session:
        output = second_session.run(pipeline_name="preprocessing")
 
    second_train_dataloader = output["train_dataloader"]
    second_test_dataloader = output["test_dataloader"]

    return second_train_dataloader, second_test_dataloader

class TestDataPreparation:
    parameters, train_dataloader, test_dataloader = run_preprocessing()
    second_train_dataloader, second_test_dataloader = second_run_preprocessing()

    def test_data_shape(self):
        train_data, _ = next(iter(self.train_dataloader))
        train_data_size = train_data.size()

        test_data, _ = next(iter(self.test_dataloader))
        test_data_size = test_data.size()
        test_size = self.test_dataloader.dataset.data.shape[0]
        
        assert np.all(
            np.array(test_data_size) == np.array([test_size, 1, 28, 28])
        ), f"test_data should have the shape[1, 1, 28, 28] but has the shape {np.array(test_data_size)}"
        assert np.all(
            np.array(train_data_size) == np.array([self.parameters["batch_size"], 1, 28, 28])
        ), f"train_data should have the shape[{self.parameters['batch_size']}, 1, 28, 28] but has the shape {np.array(train_data_size)}"


    def test_data_size(self):
        training_size = self.train_dataloader.dataset.data.shape[0]
        test_size = self.test_dataloader.dataset.data.shape[0]

        assert (
            training_size == self.parameters["TRAINING_SIZE"]
        ), f"training_size is {training_size} but should be {self.parameters['TRAINING_SIZE']}"
        assert (
            test_size == self.parameters["TEST_SIZE"]
        ), f"test_size is {test_size} but should be {self.parameters['TEST_SIZE']}"

    def test_normalization(self):
        train_data, _ = next(iter(self.train_dataloader))
        test_data, _ = next(iter(self.test_dataloader))

        assert torch.max(train_data) <= 1, "train_data is not normalized"
        assert torch.max(test_data) <= 1, "test_data is not normalized"

    def test_data_reproducability(self):
        train_data = self.train_dataloader.dataset.data
        test_data = self.test_dataloader.dataset.data
        second_train_data = self.second_train_dataloader.dataset.data
        second_test_data = self.second_test_dataloader.dataset.data

        assert torch.all(torch.eq(train_data, second_train_data)), "data preparation pipeline is not reproducable"
        assert torch.all(torch.eq(test_data, second_test_data)), "data preparation pipeline is not reproducable"
