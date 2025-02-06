import numpy as np
import torch

import scipy

from .simple import Simple


class Chebyshev(Simple):
    """Chebyshev Polynomial basis.

    The basis is optionally be transformed with a learned linear layer (``trainable=True``),
    optionally with a separate transformation per degree (``per_degree=True``).
    The target number of features is specified by ``num_features``.

    Attributes:
        cutoff (Tensor): Cutoff distance.
        max_angular (int): Maximum spherical harmonic order.
        n_per_l (list): Number of features per degree.

    """

    def __init__(self, *args, **kwargs):
        """Initialise the Chebyshev basis.

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

        n = self.num_radial
        v = np.arange(n)
        self.register_buffer("v", torch.from_numpy(v))

    def expand(self, r):
        """Compute the Chebyshev polynomial basis.

        Args:
            r (Tensor): Input distances of shape ``[pair]``.

        Returns:
            Expansion of shape ``[pair, num_radial]``.
        """
        r = r.unsqueeze(-1) / self.cutoff
        k = self.v

        mask0 = r < 0
        mask1 = r > 1
        mask = torch.logical_or(mask0, mask1)
        y = torch.where(
                mask,
                0,
                torch.cos(k * torch.arccos(2 * r - 1)))

        return y
