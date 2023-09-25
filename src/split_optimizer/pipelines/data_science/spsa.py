import torch
import numpy as np
from pennylane import SPSAOptimizer


class QNG(SPSAOptimizer, torch.optim.Optimizer):
    def __init__(self, params):
        super(torch.optim.Optimizer).__init__(params)

    def step(self, closure=None):
        loss = None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0

                p, loss = super(SPSAOptimizer).step_and_cost(closure, p)

        return loss
