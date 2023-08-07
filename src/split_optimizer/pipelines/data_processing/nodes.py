import numpy as np
import torch
from typing import Any, Dict, Tuple
from torchvision import datasets
from torchvision.transforms import ToTensor
import tensorflow as tf


def prepare_data(
    batch_size: int,
    TRAINING_SIZE: int,
    TEST_SIZE: int,
    seed: int,
) -> Dict:
    data = tf.keras.datasets.mnist.load_data(
    path="/Users/mona/Documents/quantengruppe_hiwi/split_optimizer/data/01_raw/mnist.npz")
    classes= [0,1,2,3,4,5,6,7,8,9]

    (x_train_full, y_train_full), (x_test_full, y_test_full) = data 

    x_train = x_train_full[:TRAINING_SIZE]
    x_test = x_test_full[:TEST_SIZE]

    # one-hot-encoding for the labels
    y_train = np.zeros((TRAINING_SIZE, len(classes)))
    y_test = np.zeros((TEST_SIZE, len(classes)))

    for i in range(TRAINING_SIZE):
        y_train[i, classes.index(y_train_full[i])] = 1

    for i in range(TEST_SIZE):
        y_test[i, classes.index(y_test_full[i])] = 1
    

    x_train=np.divide(x_train, 255)
    x_test=np.divide(x_test, 255)


    torch.manual_seed(seed)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test).float()


    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)

    
    train_dataloader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, shuffle=False, batch_size=1
    )
    return {"train_dataloader": train_dataloader, "test_dataloader": test_dataloader}
