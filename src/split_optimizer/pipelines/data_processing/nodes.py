import numpy as np
import torch
from torchvision.transforms import ToTensor
from split_optimizer.helpers.dataset import OneHotMNIST


def load_data():
    train_dataset = OneHotMNIST(
        root="mnist",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_dataset = OneHotMNIST(
        root="mnist",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
    }


def select_classes(train_dataset, test_dataset, classes):
    #TODO: somehow those datasets contains both train and test data
    train_class_mask = np.isin(train_dataset.targets, classes)
    test_class_mask = np.isin(test_dataset.targets, classes)

    train_dataset.targets = train_dataset.targets[train_class_mask]
    train_dataset.data = train_dataset.data[train_class_mask]

    test_dataset.targets = test_dataset.targets[test_class_mask]
    test_dataset.data = test_dataset.data[test_class_mask]

    return {
        "train_dataset_selected": train_dataset,
        "test_dataset_selected": test_dataset,
    }


def reduce_size(
    train_dataset, test_dataset, TRAINING_SIZE, TEST_SIZE
):
    train_dataset.targets = train_dataset.targets[:TRAINING_SIZE]
    train_dataset.data = train_dataset.data[:TRAINING_SIZE]
    test_dataset.targets = test_dataset.targets[:TEST_SIZE]
    test_dataset.data = test_dataset.data[:TEST_SIZE]

    return {
        "test_dataset_size_reduced": test_dataset,
        "train_dataset_size_reduced": train_dataset,
    }


def reduce_classes(
    test_dataset, train_dataset, classes, TRAINING_SIZE, TEST_SIZE
):
    # one-hot-encoding for the labels
    # y_train = torch.zeros((TRAINING_SIZE), dtype=torch.LongTensor)
    # y_test = torch.zeros((TEST_SIZE), dtype=torch.LongTensor)

    for i, c in enumerate(classes):
        train_dataset.targets[torch.where(train_dataset.targets == c)[0]] = i
        test_dataset.targets[torch.where(test_dataset.targets == c)[0]] = i

    # test_dataset_reduced.targets = y_test
    # train_dataset_reduced.targets = y_train

    return {
        "test_dataset_class_reduced": test_dataset,
        "train_dataset_class_reduced": train_dataset,
    }


def normalize(test_dataset, train_dataset):
    test_dataset.data = np.divide(test_dataset.data, test_dataset.data.max())
    train_dataset.data = np.divide(train_dataset.data, train_dataset.data.max())
    return {
        "test_dataset_normed": test_dataset,
        "train_dataset_normed": train_dataset,
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


def calculate_class_weights(train_dataset, classes, TRAINING_SIZE):
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

    class_weights_train = torch.tensor(class_weights_train, dtype=torch.float32)

    return {"class_weights_train": class_weights_train}
