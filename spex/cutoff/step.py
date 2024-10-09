import torch


class Step(torch.nn.Module):
    def __init__(self, cutoff):
        super().__init__()

        self.spec = {"cutoff": cutoff}

        self.cutoff_fn = torch.jit.trace(step(cutoff), example_inputs=torch.zeros(3))

    def forward(self, r):
        return self.cutoff_fn(r)


def step(cutoff):
    def _step(r):
        ones = torch.ones_like(r)
        zeros = torch.zeros_like(r)

        return torch.where(r < cutoff, ones, zeros)

    return _step
