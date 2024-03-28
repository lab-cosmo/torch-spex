import numpy as np
import torch


def get_cutoff_function(cutoff, spec):
    assert spec in (
        "shifted_cosine",
        "step",
    ), "we do not yet support custom cutoff functions"

    if spec == "shifted_cosine":
        fn = shifted_cosine(cutoff, width=0.5)
    elif spec == "step":
        fn = step(cutoff)

    # apparently torch doesn't properly fold in local
    # variables, so this makes things work (dubious solution)
    return torch.jit.trace(fn, example_inputs=torch.zeros(3))


def shifted_cosine(cutoff, width=0.5):
    onset = cutoff - width

    def fn(r):
        ones = torch.ones_like(r)

        zeros = torch.zeros_like(r)
        cosine = 0.5 * (1.0 + torch.cos(np.pi * (r - onset) / width))

        left_of_onset = torch.where(r < onset, ones, cosine)
        left_of_cutoff = torch.where(r < cutoff, ones, zeros)

        return left_of_onset * left_of_cutoff

    return fn


def step(cutoff):
    def fn(r):
        ones = torch.ones_like(r)
        zeros = torch.zeros_like(r)

        return torch.where(r < cutoff, ones, zeros)

    return fn
