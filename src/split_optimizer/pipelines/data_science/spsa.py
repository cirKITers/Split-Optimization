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
        self,
        params,
        argnum,
        maxiter=100,  # TODO: propagate to kedro params
        alpha=0.602,
        gamma=0.101,
        c=0.2,
        A=None,
        a=None,
    ):
        # Initialize a default dictionary, we utilize this to store any optimizer related hyperparameters
        # Note that this just follows the torch optimizer approach and is not mandatory
        defaults = dict(maxiter=maxiter, alpha=alpha, gamma=gamma, c=c, A=A, a=a)

        self.argnum = argnum

        # Initialize the QNG optimizer
        qml.SPSAOptimizer.__init__(
            self, maxiter=maxiter, alpha=alpha, gamma=gamma, c=c, A=A, a=a
        )

        # Initialize the Torch Optimizer Base Class
        torch.optim.Optimizer.__init__(self, params, defaults)

        self.requires_closure = True

    def step(self, closure, data, target, *args, **kwargs):
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

                # we detach the params from the computation graph to preserve their value
                params = p.detach()
                params.requires_grad = True

                # we can get the gradients of those parameters using the following line
                # note that in there, p gets altered (that's why we had to detach it in the line above)
                g = self.compute_grad(
                    closure, args=p, kwargs=dict(data=data, target=target)
                )

                # here we reuse the params from above and apply the calculated gradients
                p.data = torch.stack(self.apply_grad(g, params))

        # unwrap from list if one argument, cleaner return
        if len(p) == 1:
            return p[0]

        return p

    def compute_grad(self, objective_fn, args, kwargs):
        r"""Approximate the gradient of the objective function at the
        given point.

        Directly derived from the Pennylane SPSA but removed the shots stuff and switched everything to torch tensors

        Args:
            objective_fn (function): The objective function for optimization
            args (tuple): tuple of NumPy array containing the current parameters
                for objective function
            kwargs (dict): keyword arguments for the objective function

        Returns:
            tuple (array): NumPy array containing the gradient
                :math:`\hat{g}_k(\hat{\theta}_k)`
        """
        ck = self.c / self.k**self.gamma

        delta = []
        thetaplus = list(args)
        thetaminus = list(args)

        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                # Use the symmetric Bernoulli distribution to generate
                # the coordinates of delta. Note that other distributions
                # may also be used (they need to satisfy certain conditions).
                # Refer to the paper linked in the class docstring for more info.
                di = torch.bernoulli(torch.ones(arg.shape) * 0.5) * 2 - 1
                multiplier = ck * di
                thetaplus[index] = arg + multiplier
                thetaminus[index] = arg - multiplier
                delta.append(di)
        args.data = torch.stack(thetaplus)
        yplus = objective_fn(**kwargs)
        args.data = torch.stack(thetaminus)
        yminus = objective_fn(**kwargs)
        # TODO was passierte hier mit SHOTS
        grad = [(yplus - yminus) / (2 * ck * di) for di in delta]

        return tuple(grad)
