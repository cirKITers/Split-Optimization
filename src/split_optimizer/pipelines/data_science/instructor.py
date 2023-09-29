import torch.nn as nn
from typing import Dict, List

from torch.utils.data.dataloader import DataLoader

from .optimizer import SplitOptimizer, Adam, SGD


class Instructor:
    def __init__(
        self,
        model: nn.Module,
        loss_func: str,
        optimizer: List,
        epochs: int,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        class_weights_train: List,
        # Optuna
        report_callback=None,
        early_stop_callback=None,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.epochs = epochs

        if loss_func == "CrossEntropyLoss":
            self.train_loss = nn.CrossEntropyLoss(weight=class_weights_train)
            self.test_loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f"{loss_func} is not a loss function in [CrossEntropyLoss]"
            )  # TODO: shall we actually add more loss functions?

        if "combined" not in optimizer:
            self.optimizer = SplitOptimizer(model, optimizer)
        else:
            if optimizer["combined"]["name"] == "Adam":
                self.optimizer = Adam(model.parameters(), optimizer["combined"])
            elif optimizer["combined"]["name"] == "SGD":
                self.optimizer = SGD(model.parameters(), optimizer["combined"])
            else:
                raise ValueError(
                    f"{optimizer['combined']['name']} is not an optimizer in [Adam, SGD]"
                )

    def objective_function(self, data, target, train=True):
        output = self.model(data)
        if train:
            loss = self.train_loss(output, target)
        else:
            loss = self.test_loss(output, target)
        return loss
