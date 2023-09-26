import torch
import pennylane as qml


class SPSA(qml.SPSAOptimizer, torch.optim.Optimizer):
    """Implementation of the Quantum Natural Gradient Optimizer

    Args:
        params: Parameters of the torch model
        qnode: QNode instance that is being used in the model
        lr: Step size/ learning rate of the optimizer
        dampening: Float for metric tensor regularization
    """

    def __init__(
        self, params, qnode, argnum, maxiter=None, alpha=0.602, gamma=0.101, c=0.2, A=None, a=None
    ):
        # Initialize a default dictionary, we utilize this to store any optimizer related hyperparameters
        # Note that this just follows the torch optimizer approach and is not mandatory
        defaults = dict(maxiter=maxiter, alpha=alpha, gamma=gamma, c=c, A=A, a=a)

        self.argnum = argnum

        # Initialize the QNG optimizer
        qml.SPSAOptimizer.__init__(self, maxiter=maxiter, alpha=alpha, gamma=gamma, c=c, A=A, a=a)

        # Initialize the Torch Optimizer Base Class
        torch.optim.Optimizer.__init__(self, params, defaults)

        self.requires_closure = True

    def step(self, closure=None, *args, **kwargs):
        """Step method implementation. We call this to update the parameter values

        Args:
            closure (Callable, optional): The closure is a helper fn that is being called by the optimizer to get the metric tensor for the current parameter configuration. Defaults to None.

        Returns:
            Tensor: Updated parameters
        """

        # Iteratate the parameter groups (i.e. parts of the model (just one in this case))
        # We obtain the param_groups variable after the torch optimizer instantiation
        for pg in self.param_groups:
            # Each group is a dictionary where the actual parameters can be accessed using the "params" key
            for p in pg["params"]:
                # p is now a set of parameters (i.e. the weights of the VQC)

                # we can get the gradients of those parameters using the following line
                g = self.compute_grad(closure, args, kwargs)


                p.data = torch.tensor(
                    self.apply_grad(g.detach().numpy(), p.detach().numpy()),
                    requires_grad=True,
                )

        # unwrap from list if one argument, cleaner return
        if len(p) == 1:
            return p[0]

        return p
