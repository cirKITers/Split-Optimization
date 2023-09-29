import torch.nn as nn
from typing import Dict, List
from .optimizer import initialize_optimizer

from torch.utils.data.dataloader import DataLoader


class Instructor:
    def __init__(
        self,
        model: nn.Module,
        loss_func: str,
        learning_rate: float,
        optimizer_list: List,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        class_weights_train: List,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        if loss_func == "CrossEntropyLoss":
            self.train_loss = nn.CrossEntropyLoss(weight=class_weights_train)
            self.test_loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f"{loss_func} is not a loss function in [CrossEntropyLoss]"
            )  # TODO: shall we actually add more loss functions?

        self.optimizer = initialize_optimizer(model, learning_rate, optimizer_list)

    def objective_function(self, data, target, train=True):
        output = self.model(data)
        if train:
            loss = self.train_loss(output, target)
        else:
            loss = self.test_loss(output, target)
        return loss
