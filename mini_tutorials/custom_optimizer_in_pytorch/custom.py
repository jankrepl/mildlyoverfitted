import numpy as np
import torch
from torch.optim import Optimizer

class WeirdDescent(Optimizer):
    """Take a coordinate descent step for a random parameter.

    And also, make every 100th step way bigger.
    """
    def __init__(self, parameters, lr=1e-3):
        defaults = {"lr": lr}
        super().__init__(parameters, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if not self.state:
            self.state["step"] = 1
        else:
            self.state["step"] += 1

        c = 1
        if self.state["step"] % 100 == 0:
            c = 100

        grad = None
        while grad is None:
            param_group = np.random.choice(self.param_groups)
            tensor = np.random.choice(param_group["params"])
            grad = tensor.grad.data

        element_ix = np.random.randint(tensor.numel())

        mask_flat = torch.zeros(tensor.numel())
        mask_flat[element_ix] = 1
        mask = mask_flat.reshape(tensor.shape)

        tensor.data.add_(grad * mask, alpha=-param_group["lr"] * c)

        return loss
