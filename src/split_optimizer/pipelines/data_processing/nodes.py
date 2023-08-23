import numpy as np
import torch
from typing import Any, Dict, Tuple
import tensorflow as tf


def load_data():
    data = tf.keras.datasets.mnist.load_data()
    # daten sind bereits geshuffelt
    (x_train_full, y_train_full), (x_test_full, y_test_full) = data

    return {
        "x_train_full": x_train_full,
        "y_train_full": y_train_full,
        "x_test_full": x_test_full,
        "y_test_full": y_test_full,
    }


def format_data(
    x_train_full: np.array,
    y_train_full: np.array,
    x_test_full: np.array,
    y_test_full: np.array,
    TRAINING_SIZE: int,
    TEST_SIZE: int,
    number_classes: int,
):
    classes = range(number_classes)

    # reduce number of samples
    x_train = x_train_full[:TRAINING_SIZE]
    x_test = x_test_full[:TEST_SIZE]

    # one-hot-encoding for the labels
    y_train = np.zeros((TRAINING_SIZE, number_classes))
    y_test = np.zeros((TEST_SIZE, number_classes))

    for i in range(TRAINING_SIZE):
        y_train[i, classes.index(y_train_full[i])] = 1

    for i in range(TEST_SIZE):
        y_test[i, classes.index(y_test_full[i])] = 1

    # normalization
    x_train = np.divide(x_train, 255)
    x_test = np.divide(x_test, 255)
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


def create_dataloader(
    x_train: np.array,
    y_train: np.array,
    x_test: np.array,
    y_test: np.array,
    seed: int,
    batch_size: int,
):
    # set seed
    torch.manual_seed(seed)

    # convert to tensor
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test).float()

    # add channel for training
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])

    # create Tensor Dataset
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)

    # create Data Loader
    train_dataloader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, shuffle=False, batch_size=1
    )
    return {"train_dataloader": train_dataloader, "test_dataloader": test_dataloader}


def calculate_class_weights(
    y_train_full: np.array,
    number_classes: int,
    TRAINING_SIZE: int,
):
    y_train = y_train_full[:TRAINING_SIZE]
    class_weights_train = np.array(())
    for i in range(number_classes):
        train_elements = np.where(y_train == i)
        class_weights_train = np.append(
            class_weights_train,
            np.divide(TRAINING_SIZE, train_elements[0].size * number_classes),
        )
        # class weight formula source:
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

    class_weights_train = torch.from_numpy(class_weights_train)

    return {"class_weights_train": class_weights_train}
