import numpy as np
import torch

import scipy

from .simple import Simple


class Gaussian(Simple):
    """
    Gaussian radial basis functions.
    
    The Gaussian basis functions are defined as ``exp(-gamma * (r - k * l / (K - 1)) ** 2)``,
    where ``gamma = sqrt(2 * K) * (K - 1)``, l is the cutoff distance and ``k`` is the degree 
    of the basis.
        
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
        """Initialise the Gaussian basis.

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
        
        K = torch.tensor(self.num_radial)     
        gamma = torch.sqrt(2 * K) * (K - 1)
        
        self.register_buffer("K", K)
        self.register_buffer("gamma", gamma)

    def expand(self, r):
        """Compute the Bernstein polynomial basis.

        Args:
            r (Tensor): Input distances of shape ``[pair]``.

        Returns:
            Expansion of shape ``[pair, num_radial]``.
        """

        r = r.unsqueeze(-1)
        gamma = self.gamma
        K = self.K
        k = torch.arange(K, dtype=torch.float32)
        y = torch.exp(-gamma/self.cutoff * (r - (k * self.cutoff) / (K - 1)) ** 2)
        
        return y
        
        
