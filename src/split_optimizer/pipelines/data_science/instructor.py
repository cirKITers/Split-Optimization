import torch
import torch.nn as nn
from typing import Dict, List

from torch.utils.data.dataloader import DataLoader
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_auroc, multiclass_f1_score
from .optimizer import SplitOptimizer, Adam, SGD, NGD


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
        torch_seed:int,
        # Optuna
        report_callback=None,
        early_stop_callback=None,
    ) -> None:
        if torch_seed is not None:
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(torch_seed)

        self.model = model

        # trigger manual weight init as we cannot guarantee that torch seeding ran previously
        self.model.reset_parameters()

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.epochs = epochs

        if "combined" not in optimizer:
            self.optimizer = SplitOptimizer(model, optimizer)
        else:
            opt_name = optimizer["combined"]["name"]
            del optimizer["combined"]["name"]

            if opt_name == "Adam":
                self.optimizer = Adam(model.parameters(), **optimizer["combined"])
            elif opt_name == "SGD":
                self.optimizer = SGD(model.parameters(), **optimizer["combined"])
            elif opt_name == "NGD":
                self.optimizer = NGD(model.parameters(), **optimizer["combined"])
            else:
                raise ValueError(f"{opt_name} is not an optimizer in [Adam, SGD]")

        num_classes = len(
            class_weights_train
        )  # TODO: infering this could be risky, but currently, I cannot imagine a scenario where this wouldn't match

        # this dictionary contains the available metrics as functionals (f) as well as any additional arguments that they might need for training and evaluation cases
        # furthermore a 'sign' (s) is provided, that is multiplied with the value when the metric is being used as a loss
        self.metrics = {
            "CrossEntropy": {
                "f": nn.functional.cross_entropy,
                "train_kwargs": dict(weight=class_weights_train),
                "eval_kwargs": dict(),
                "s": 1,
            },
            "Accuracy": {
                "f": multiclass_accuracy,
                "train_kwargs": dict(num_classes=num_classes),
                "eval_kwargs": dict(num_classes=num_classes),
                "s": -1,
            },
            "AUROC": {
                "f": multiclass_auroc,
                "train_kwargs": dict(num_classes=num_classes),
                "eval_kwargs": dict(num_classes=num_classes),
                "s": -1,
            },
            "F1": {
                "f": multiclass_f1_score,
                "train_kwargs": dict(num_classes=num_classes),
                "eval_kwargs": dict(num_classes=num_classes),
                "s": -1,
            },
        }
        self.loss_func = loss_func

        if loss_func not in self.metrics.keys():
            raise KeyError(f"No loss {loss_func} in {self.metrics}")

    def objective_function(self, data, target):
        output = self.model(data)

        metrics_val = {}
        loss_val = torch.nan
        for name, metric in self.metrics.items():
            kwargs = (
                metric["train_kwargs"] if self.model.training else metric["eval_kwargs"]
            )

            if name == self.loss_func:
                loss_val = metric["s"] * metric["f"](output, target, **kwargs)
            else:
                # we don't apply the sign here as it is only relevant for optimization and could be confusing when a negative accuracy or so is displayed
                metrics_val[name] = metric["f"](output, target, **kwargs)

        return output, loss_val, metrics_val
