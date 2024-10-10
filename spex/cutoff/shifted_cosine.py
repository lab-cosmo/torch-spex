import numpy as np
import torch


class ShiftedCosine(torch.nn.Module):
    def __init__(self, cutoff, width=0.5):
        super().__init__()

        self.spec = {"cutoff": cutoff, "width": width}

        self.cutoff_fn = torch.jit.trace(
            shifted_cosine(cutoff, width=width), example_inputs=torch.zeros(3)
        )

    def forward(self, r):
        return self.cutoff_fn(r)


def shifted_cosine(cutoff, width=0.5):
    def fn(r):
        onset = cutoff - width
        ones = torch.ones_like(r)

        zeros = torch.zeros_like(r)
        cosine = 0.5 * (1.0 + torch.cos(np.pi * (r - onset) / width))

        left_of_onset = torch.where(r < onset, ones, cosine)
        left_of_cutoff = torch.where(r < cutoff, ones, zeros)

        return left_of_onset * left_of_cutoff

    return fn
