import torch.optim as optim


class SplitOptimizer:
    def __init__(self, model, lr):
        self.classical_optimizer = optim.Adam(model.clayer.parameters(), lr)
        self.quantum_optimizer = optim.Adam(model.qlayer.parameters(), lr)

    def zero_grad(self):
        self.classical_optimizer.zero_grad()
        self.quantum_optimizer.zero_grad()

    def step(self):
        self.classical_optimizer.step()
        self.quantum_optimizer.step()
