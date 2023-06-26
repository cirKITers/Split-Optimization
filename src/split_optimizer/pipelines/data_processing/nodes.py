import numpy as np
import torch
from typing import Dict


def prepare_data(
    train_filepath: str,
    test_filepath: str,
    batch_size: int,
    TRAINING_SIZE: int,
    TEST_SIZE: int,
    seed: int,
) -> Dict:

    train_data = np.load(train_filepath)  # enthält 7500 samples
    test_data = np.load(test_filepath)  # enthält 1500 samples

    x_train = train_data["features"][:TRAINING_SIZE]
    # y_train = train_data["labels"][:TRAINING_SIZE]
    x_test = test_data["features"][:TEST_SIZE]
    # y_test = train_data["labels"][:TEST_SIZE]

    # one-hot-encoding for the labels
    y_train = np.zeros((TRAINING_SIZE, len(train_data["classes"])))
    y_test = np.zeros((TEST_SIZE, len(train_data["classes"])))

    for i in range(TRAINING_SIZE):
        y_train[i, list(train_data["classes"]).index(train_data["labels"][i])] = 1

    for i in range(TEST_SIZE):
        y_test[i, list(test_data["classes"]).index(test_data["labels"][i])] = 1

    torch.manual_seed(seed)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test).float()

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=1
    )
    return {"train_dataloader": train_dataloader, "test_dataloader": test_dataloader}
