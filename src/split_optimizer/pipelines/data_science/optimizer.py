import torch.optim as optim
from .ngd import NGD
from .qng import QNG
from .spsa import SPSA


class SplitOptimizer:
    def __init__(self, model, optimizer):
        classical_opt_name = optimizer["classical"]["name"]
        quantum_opt_name = optimizer["quantum"]["name"]

        del optimizer["classical"]["name"], optimizer["quantum"]["name"]

        if classical_opt_name == "Adam":
            self.classical_optimizer = Adam(
                model.clayer.parameters(), **optimizer["classical"]
            )
        elif classical_opt_name == "SGD":
            self.classical_optimizer = SGD(
                model.clayer.parameters(), **optimizer["classical"]
            )
        elif classical_opt_name == "NGD":
            self.classical_optimizer = NGD(
                model.clayer.parameters(), **optimizer["classical"]
            )
        else:
            raise ValueError(
                f"{classical_opt_name} is not an optimizer for the classical part in [Adam, SGD]"
            )

        if quantum_opt_name == "NGD":
            self.quantum_optimizer = NGD(
                model.qlayer.parameters(), **optimizer["quantum"]
            )
        elif quantum_opt_name == "QNG":
            self.quantum_optimizer = QNG(
                model.qlayer.parameters(),
                model.qnode,
                model.vqc.argnum,
                **optimizer["quantum"],
            )
        elif quantum_opt_name == "SPSA":
            self.quantum_optimizer = SPSA(
                model.qlayer.parameters(), model.vqc.argnum, **optimizer["quantum"]
            )
        elif quantum_opt_name == "Adam":
            self.quantum_optimizer = Adam(
                model.qlayer.parameters(), **optimizer["quantum"]
            )
        elif quantum_opt_name == "SGD":
            self.quantum_optimizer = SGD(
                model.qlayer.parameters(), **optimizer["quantum"]
            )
        else:
            raise ValueError(
                f"{quantum_opt_name} is not an optimizer for the quantum part in [Adam, SGD, NGD, QNG, SPSA]"
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
        kwargs.setdefault(
            "momentum", 0.9
        )  # actually important as not converging otherwise
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
