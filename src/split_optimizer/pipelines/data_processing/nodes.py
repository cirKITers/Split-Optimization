import numpy as np
import torch
from torchvision.transforms import ToTensor
from split_optimizer.helpers.dataset import OneHotMNIST


def load_data():
    train_dataset_full = OneHotMNIST(
        root="mnist",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_dataset_full = OneHotMNIST(
        root="mnist",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return {
        "train_dataset_full": train_dataset_full,
        "test_dataset_full": test_dataset_full,
    }


def select_classes(train_dataset_full, test_dataset_full, classes):
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
    train_dataset_selected, test_dataset_selected, TRAINING_SIZE, TEST_SIZE
):
    train_dataset_selected.targets = train_dataset_selected.targets[:TRAINING_SIZE]
    train_dataset_selected.data = train_dataset_selected.data[:TRAINING_SIZE]
    test_dataset_selected.targets = test_dataset_selected.targets[:TEST_SIZE]
    test_dataset_selected.data = test_dataset_selected.data[:TEST_SIZE]

    return {
        "test_dataset_reduced": test_dataset_selected,
        "train_dataset_reduced": train_dataset_selected,
    }


def onehot(
    test_dataset_reduced, train_dataset_reduced, classes, TRAINING_SIZE, TEST_SIZE
):
    # one-hot-encoding for the labels
    y_train = torch.zeros((TRAINING_SIZE, len(classes)))
    y_test = torch.zeros((TEST_SIZE, len(classes)))

    for i, c in enumerate(classes):
        y_train[torch.where(train_dataset_reduced.targets == c)[0], i] = 1
        y_test[torch.where(test_dataset_reduced.targets == c)[0], i] = 1

    test_dataset_reduced.targets = y_test
    train_dataset_reduced.targets = y_train

    return {
        "test_dataset_onehot": test_dataset_reduced,
        "train_dataset_onehot": train_dataset_reduced,
    }


def normalize(test_dataset_onehot, train_dataset_onehot):
    test_dataset_onehot.data = np.divide(test_dataset_onehot.data, test_dataset_onehot.data.max())
    train_dataset_onehot.data = np.divide(train_dataset_onehot.data, train_dataset_onehot.data.max())
    return {
        "test_dataset": test_dataset_onehot,
        "train_dataset": train_dataset_onehot,
    }


def create_dataloader(
    train_dataset,
    test_dataset,
    seed: int,
    batch_size: int,
):
    # set seed
    torch.manual_seed(seed)
    # create Data Loader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=1
    )
    return {"train_dataloader": train_dataloader, "test_dataloader": test_dataloader}


def calculate_class_weights(train_dataset_reduced, classes, TRAINING_SIZE):
    y_train = train_dataset_reduced.targets
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
