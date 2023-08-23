import torch.optim as optim
from .ngd import NGD

def initialize_optimizer(model, lr, optimizer_list):
    if len(optimizer_list) == 2:
        return SplitOptimizer(model, lr, optimizer_list)
    elif optimizer_list[0] == "Adam":
        return Adam(model.parameters(), lr)
    elif optimizer_list[0] == "SGD":
        return SGD(model.parameters(), lr) #TODO: Add momentum as Kedro parameter
    else:
        raise ValueError(f"{optimizer_list} is not an optimizer in [Adam, SGD]")


class SplitOptimizer:
    def __init__(self, model, lr, optimizer_list):
        if optimizer_list[0] == "Adam":
            self.classical_optimizer = Adam(model.clayer.parameters(), lr)
        elif optimizer_list[0] == "SGD":
            momentum = 0.9
            self.classical_optimizer = SGD(model.clayer.parameters(), lr, momentum)
        else:
            raise ValueError(
                f"{optimizer_list[0]} is not an optimizer for the classical part in [Adam, SGD]"
            )

        if optimizer_list[1] == "NGD":
            self.quantum_optimizer = NGD(model.qlayer.parameters(), lr)
        elif optimizer_list[1] == "QNG":
            self.quantum_optimizer = QNG(model.qlayer.parameters(), lr)
        elif optimizer_list[1] == "SPSA":
            self.quantum_optimizer = SPSA(model.qlayer.parameters(), lr)
        elif optimizer_list[1] == "Adam":
            self.quantum_optimizer = Adam(model.qlayer.parameters(), lr)
        elif optimizer_list[1] == "SGD":
            momentum = 0.9
            self.quantum_optimizer = SGD(model.qlayer.parameters(), lr, momentum)
        else:
            raise ValueError(
                f"{optimizer_list[1]} is not an optimizer for the quantum part in [Adam, SGD, NGD, QNG, SPSA]"
            )

    def zero_grad(self):
        self.classical_optimizer.zero_grad()
        self.quantum_optimizer.zero_grad()

    def step(self):
        self.classical_optimizer.step()
        self.quantum_optimizer.step()


class Adam(optim.Adam):
    def __init__(self, model_params, lr):
        super(Adam, self).__init__(model_params, lr)


class SGD(optim.SGD):
    def __init__(self, model_params, lr, momentum=0.9):
        super(SGD, self).__init__(model_params, lr, momentum)


class NGD(NGD):
    def __init__(self, model_params, lr, momentum=0.9, dampening=0, weight_decay=0, nesterov=False):
        super(NGD, self).__init__(model_params, lr, momentum, dampening, weight_decay, nesterov)


class QNG:
    def __init__(self, model_params, lr):
        raise NotImplementedError("QNG is not implemented yet")


class SPSA:
    def __init__(self, model_params, lr):
        raise NotImplementedError("SPSA is not implemented yet")
