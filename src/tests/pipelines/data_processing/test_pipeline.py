from pathlib import Path
import numpy as np
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
import torch

def test_data_preparation():

    bootstrap_project(Path.cwd())
    with KedroSession.create() as session:
        output = session.run(pipeline_name="preprocessing")

    parameters = session.load_context().config_loader["parameters"]["data_processing"]

    train_dataloader = output["train_dataloader"]
    test_dataloader = output["test_dataloader"]

    

    batch_size_train = train_dataloader.batch_size

    train_dataset = train_dataloader.dataset.data
    test_dataset = test_dataloader.dataset.data

    training_size = train_dataset.shape[0]
    test_size = test_dataset.shape[0]

    train_data, _ = next(iter(train_dataloader))
    train_data_size = train_data.size()

    test_data, _ = next(iter(test_dataloader))
    test_data_size = test_data.size()

    # test normalization
    assert torch.max(train_data) <= 1, "train_data is not normalized"
    assert torch.max(test_data) <= 1, "test_data is not normalized"
   
    # test shape
    assert np.all(
        np.array(test_data_size) == np.array([test_size, 1, 28, 28])
    ), f"test_data should have the shape[1, 1, 28, 28] but has the shape {np.array(test_data_size)}"
    assert np.all(
        np.array(train_data_size) == np.array([parameters["batch_size"], 1, 28, 28])
    ), f"train_data should have the shape[{parameters['batch_size']}, 1, 28, 28] but has the shape {np.array(train_data_size)}"
    assert batch_size_train == parameters["batch_size"]
    
    #test size of the datasets
    assert (
        training_size == parameters["TRAINING_SIZE"]
    ), f"training_size is {training_size} but should be {parameters['TRAINING_SIZE']}"
    assert (
        test_size == parameters["TEST_SIZE"]
    ), f"test_size is {test_size} but should be {parameters['TEST_SIZE']}"


    #check reproducability 
    session.close()
    bootstrap_project(Path.cwd())
    with KedroSession.create() as session:
        output = session.run(pipeline_name="preprocessing")


    second_train_dataloader = output["train_dataloader"]
    second_test_dataloader = output["test_dataloader"]
    second_train_data = second_train_dataloader.dataset.data
    second_test_data = second_test_dataloader.dataset.data


    assert torch.all(torch.eq(train_dataset, second_train_data))
    assert torch.all(torch.eq(test_dataset, second_test_data))