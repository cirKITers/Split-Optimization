import torch.optim as optim
from .ngd import NGD
from .qng import QNG
from .spsa import SPSA


def initialize_optimizer(model, lr, optimizer_list):
    if len(optimizer_list) == 2:
        return SplitOptimizer(model, lr, optimizer_list)
    elif optimizer_list[0] == "Adam":
        return Adam(model.parameters(), lr)
    elif optimizer_list[0] == "SGD":
        return SGD(model.parameters(), lr)  # TODO: Add momentum as Kedro parameter
    else:
        raise ValueError(f"{optimizer_list} is not an optimizer in [Adam, SGD]")


class SplitOptimizer:
    def __init__(self, model, lr, optimizer_list):
        if optimizer_list[0] == "Adam":
            self.classical_optimizer = Adam(model.clayer.parameters(), lr)
        elif optimizer_list[0] == "SGD":
            self.classical_optimizer = SGD(model.clayer.parameters(), lr)
        elif optimizer_list[0] == "NGD":
            self.classical_optimizer = NGD(model.clayer.parameters(), lr)
        else:
            raise ValueError(
                f"{optimizer_list[0]} is not an optimizer for the classical part in [Adam, SGD]"
            )

        if optimizer_list[1] == "NGD":
            self.quantum_optimizer = NGD(model.qlayer.parameters(), lr)
        elif optimizer_list[1] == "QNG":
            self.quantum_optimizer = QNG(
                model.qlayer.parameters(), model.qnode, model.vqc.argnum, lr
            )
        elif optimizer_list[1] == "SPSA":
            self.quantum_optimizer = SPSA(
                model.qlayer.parameters(), model.vqc.argnum, lr
            )
        elif optimizer_list[1] == "Adam":
            self.quantum_optimizer = Adam(model.qlayer.parameters(), lr)
        elif optimizer_list[1] == "SGD":
            self.quantum_optimizer = SGD(model.qlayer.parameters(), lr)
        else:
            raise ValueError(
                f"{optimizer_list[1]} is not an optimizer for the quantum part in [Adam, SGD, NGD, QNG, SPSA]"
            )

    def zero_grad(self):
        self.classical_optimizer.zero_grad()
        self.quantum_optimizer.zero_grad()

    def step(self, data, target, qclosure=None, cclosure=None):
        self.classical_optimizer.step(cclosure)
        self.quantum_optimizer.step(qclosure, data, target)


class Adam(optim.Adam):
    def __init__(self, *args, **kwargs):
        super(Adam, self).__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        return super().step()


class SGD(optim.SGD):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("momentum", 0.9)
        super(SGD, self).__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        return super().step()


class NGD(NGD):
    def __init__(self, *args, **kwargs):
        super(NGD, self).__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        return super().step()


class QNG(QNG):
    def __init__(self, *args, **kwargs):
        super(QNG, self).__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        return super().step()


class SPSA(SPSA):
    def __init__(self, *args, **kwargs):
        super(SPSA, self).__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)
