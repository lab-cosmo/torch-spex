import torch


class Step(torch.nn.Module):
    spec = {}

    def __init__(self):
        super().__init__()

        self.cutoff_fn = torch.jit.trace(
            step, example_inputs=(torch.zeros(3), torch.tensor(1.0))
        )

    def forward(self, r):
        return self.cutoff_fn(r)


def step(r, cutoff):
    ones = torch.ones_like(r)
    zeros = torch.zeros_like(r)

    return torch.where(r < cutoff, ones, zeros)
