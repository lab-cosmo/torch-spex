import numpy as np
import torch

import scipy

from .simple import Simple


class Bernstein(Simple):
    """Bernstein Polynomial basis.

    The Bernstein polynomials are a set of orthogonal polynomials over the interval
    [0, 1]. We scale the input r to be in this interval, and then compute the polynomials
    up to degree ``num_radial-1`` (since degrees are counted from zero).

    The polynomials are defined as ``B_{n, v}(x) = C(n, v) x^v (1 - x)^(n - v)``, where
    ``C(n, v)`` is the binomial coefficient.

    The basis is optionally be transformed with a learned linear layer (``trainable=True``),
    optionally with a separate transformation per degree (``per_degree=True``).
    The target number of features is specified by ``num_features``.

    Attributes:
        cutoff (Tensor): Cutoff distance.
        max_angular (int): Maximum spherical harmonic order.
        n_per_l (list): Number of features per degree.

    """

    # implementation heavily inspired by e3x. thanks!

    def __init__(self, *args, **kwargs):
        """Initialise the Bernstein basis.

        Args:
            cutoff (float): Cutoff distance.
            num_radial (int): Number of radial basis functions.
            max_angular (int): Maximum spherical harmonic order.
            trainable (bool, optional): Whether a learned linear transformation is
                applied.
            per_degree (bool, optional): Whether to have a separate learned transform
                per degree.
            num_features (int, optional): Target number of features for learned
                transformation. Defaults to ``num_radial``.

        """
        super().__init__(*args, **kwargs)

        n = self.num_radial - 1
        v = np.arange(n + 1)
        binomln = -scipy.special.betaln(1 + n - v, 1 + v) - np.log(n + 1)

        self.register_buffer("binomln", torch.from_numpy(binomln))
        self.register_buffer("v", torch.from_numpy(v))
        self.n = n
        self.eps = torch.finfo(
            torch.float32
        ).eps  # torchscript won't let us do this at runtime

    def expand(self, r):
        """Compute the Bernstein polynomial basis.

        Args:
            r (Tensor): Input distances of shape ``[pair]``.

        Returns:
            Expansion of shape ``[pair, num_radial]``.
        """

        r = r.unsqueeze(-1) / self.cutoff
        n = self.n
        v = self.v

        mask0 = r < self.eps
        mask1 = r > 1 - self.eps
        mask = torch.logical_or(mask0, mask1)
        safe_r = torch.where(mask, 0.5, r)
        y = torch.where(
            mask,
            0,
            torch.exp(
                self.binomln + v * torch.log(safe_r) + (n - v) * torch.log1p(-safe_r)
            ),
        )
        y = torch.where(torch.logical_and(mask0, v == 0), 1, y)  # Entries for r = 0.
        y = torch.where(torch.logical_and(mask1, v == n), 1, y)  # Entries for r = 1.

        return y  # -> [pair, num_radial]
