from kedro.runner import SequentialRunner
from pathlib import Path

from kedro.io.data_catalog import DataCatalog

import numpy as np
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


from kedro.framework.project import settings


def test_data_shape():

    bootstrap_project(Path.cwd())
    with KedroSession.create() as session:
        output = session.run(pipeline_name="data_processing_pipeline")

    parameters = session.load_context().config_loader["parameters"]["data_processing"]

    train_dataloader = output["train_dataloader"]
    test_dataloader = output["test_dataloader"]

    test_dataset = test_dataloader.dataset
    train_dataset = train_dataloader.dataset

    training_size = train_dataset.data.shape[0]
    test_size = test_dataset.data.shape[0]

    batch_size_test = test_dataloader.batch_size
    batch_size_train = train_dataloader.batch_size

    train_data = train_dataloader.dataset.data
    test_data = test_dataloader.dataset.data

    train_data, _ = next(iter(train_dataloader))
    train_data_size = train_data.size()

    test_data, _ = next(iter(test_dataloader))
    test_data_size = test_data.size()

    for i in train_data[0,0]:
        for p in i:
            assert p <= 1, "train_data is not normalized"
    for i in test_data[0,0]:
        for p in i:
            assert p <= 1, "train_data is not normalized"

    assert np.all(
        np.array(test_data_size) == np.array([1, 1, 28, 28])
    ), f"test_data should have the shape[1, 1, 28, 28] but has the shape {np.array(test_data_size)}"
    assert np.all(
        np.array(train_data_size) == np.array([parameters['batch_size'], 1, 28, 28])
    ), f"train_data should have the shape[{parameters['batch_size']}, 1, 28, 28] but has the shape {np.array(train_data_size)}"
    assert batch_size_train == parameters['batch_size']
    assert (
        training_size == parameters["TRAINING_SIZE"]
    ), f"training_size is {training_size} but should be {parameters['TRAINING_SIZE']}"
    assert (
        test_size == parameters["TEST_SIZE"]
    ), f"test_size is {test_size} but should be {parameters['TEST_SIZE']}"


