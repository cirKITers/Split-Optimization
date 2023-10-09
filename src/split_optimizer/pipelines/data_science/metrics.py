import torch.nn as nn
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_auroc,
    multiclass_f1_score,
)

metrics = {
    "CrossEntropy": {
        "f": nn.functional.cross_entropy,
        "s": 1,
    },
    "Accuracy": {
        "f": multiclass_accuracy,
        "s": -1,
    },
    "AUROC": {
        "f": multiclass_auroc,
        "s": -1,
    },
    "F1": {
        "f": multiclass_f1_score,
        "s": -1,
    },
}
