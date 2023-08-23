import torch
import numpy as np


class QNG(torch.optim.Optimizer):

    def __init__(self, params, closure, lr=0.01, diag_approx=False, lam=0):
        defaults = dict(lr=0.01, diag_approx=False, lam=0)
        super().__init__(params, defaults)

        self.closure = closure


    def step(self):
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

                loss, g = self.closure(self, p.data)
                g = g.numpy()
                g += group["lam"] * np.identity(g.shape[0])

                state["step"] += 1

                d_p = torch.tensor(-group['lr'] * np.linalg.solve(g, grad))
                p.data.add_(d_p)

        return loss
