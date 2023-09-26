import numpy as np
import torch
from typing import Any, Dict, Tuple
import tensorflow as tf
import torchvision
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def imshow(sample_element, shape=(28, 28)):
    plt.imshow(sample_element[0].numpy().reshape(shape), cmap="gray")
    plt.title("Label = " + str(sample_element[1]))
    plt.show()


def load_data():

    train_dataset_full = torchvision.datasets.MNIST(
        root="mnist",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_dataset_full = torchvision.datasets.MNIST(
        root="mnist",
        train=False,
        download=True,
        transform=ToTensor(),
    )
   
    return {
        "train_dataset_full": train_dataset_full,
        "test_dataset_full": test_dataset_full
    }

def select_classes(
        train_dataset_full,
        test_dataset_full,
        classes
):
    
    
    train_class_mask = np.isin(train_dataset_full.targets, classes)
    test_class_mask = np.isin(test_dataset_full.targets, classes)

    train_dataset_full.targets = train_dataset_full.targets[train_class_mask]
    train_dataset_full.data = train_dataset_full.data[train_class_mask]

    test_dataset_full.targets = test_dataset_full.targets[test_class_mask]
    test_dataset_full.data = test_dataset_full.data[test_class_mask]

    return {
        "train_dataset_selected": train_dataset_full,
        "test_dataset_selected": test_dataset_full,
    }

def reduce_size(
        train_dataset_selected,
        test_dataset_selected,
        TRAINING_SIZE,
        TEST_SIZE
):
    """ test_dataset = torch.utils.data.Subset(test_dataset_selected, torch.arange(TEST_SIZE))
    train_dataset = torch.utils.data.Subset(
        train_dataset_selected, torch.arange(TRAINING_SIZE)
    ) """
    train_dataset_selected.targets = train_dataset_selected.targets[:TRAINING_SIZE]
    train_dataset_selected.data = train_dataset_selected.data[:TRAINING_SIZE]
    test_dataset_selected.targets = test_dataset_selected.targets[:TEST_SIZE]
    test_dataset_selected.data = test_dataset_selected.data[:TEST_SIZE]

    return {
        "test_dataset": test_dataset_selected,
        "train_dataset": train_dataset_selected,
    }


def create_dataloader(
    train_dataset,
    test_dataset,
    seed: int,
    batch_size: int,
):
    # set seed
    torch.manual_seed(seed)
    """ 
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
    """

    # create Data Loader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=1
    )
    return {"train_dataloader": train_dataloader, "test_dataloader": test_dataloader}


def calculate_class_weights(
    train_dataset,
    classes,
    TRAINING_SIZE
):
    classes = np.array(classes)
    classes = classes.reshape(len(classes),1)
    classes_onehot = OneHotEncoder(sparse=False).fit_transform(classes)
    y_train = train_dataset.targets
    class_weights_train = np.array(())
    for i in classes:
        train_elements = np.where(y_train == i)
        class_weights_train = np.append(
            class_weights_train,
            np.divide(TRAINING_SIZE, train_elements[0].size * len(classes)),
        )
        # class weight formula source:
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

    class_weights_train = torch.from_numpy(class_weights_train)

    return {"class_weights_train": class_weights_train}



""" def format_data(
    x_train_full: np.array,
    y_train_full: np.array,
    x_test_full: np.array,
    y_test_full: np.array,
    TRAINING_SIZE: int,
    TEST_SIZE: int,
    number_classes: int,
):
    classes = range(number_classes)

    # reduce to samples from the chosen classes
    train_class_mask = np.isin(y_train_full, classes)
    test_class_mask = np.isin(y_test_full, classes)

    y_train_selected = y_train_full[train_class_mask]
    x_train_selected = x_train_full[train_class_mask]

    y_test_selected = y_test_full[test_class_mask]
    x_test_selected = x_test_full[test_class_mask]

    # reduce number of samples
    x_train = x_train_selected[:TRAINING_SIZE]
    x_test = x_test_selected[:TEST_SIZE]

    # one-hot-encoding for the labels
    y_train = np.zeros((TRAINING_SIZE, number_classes))
    y_test = np.zeros((TEST_SIZE, number_classes))

    for c in classes:
        y_train[np.where(y_train_selected[:TRAINING_SIZE] == c)[0], c] = 1

    for c in classes:
        y_test[np.where(y_test_selected[:TEST_SIZE] == c)[0], c] = 1

    # normalization
    x_train = np.divide(x_train, 255)
    x_test = np.divide(x_test, 255)
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test} """


""" train_dataset_full = torchvision.datasets.MNIST(
        root="mnist",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(
                0, torch.tensor(y), value=1
            )
        ),
    )
 """