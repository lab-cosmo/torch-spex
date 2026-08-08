import numpy as np
import torch

import scipy

from .simple import Simple


class Jacobi(Simple):
    """Jacobi polynomial basis.

    It computes the Jacobi polynomial basis up to a given cutoff distance.
    The input distances are scaled to the interval [-1,1] by means of a map induced
    by the cosine function. The Jacobi polynomials are then evaluated at these.
    The Jacobi polynomials depend on two parameters ``alpha`` and ``beta``, both real
    numbers and both greater than -1. 

    Specific choices of alpha and beta gives different polynomials:
    alpha = beta = 0 gives the Legendre polynomials
    alpha = beta = 1 gives the Chebyshev polynomials of the first kind
    alpha = beta = s/2 gives the Gegenbaur polynomials of order s (with s integer)

    The basis is optionally be transformed with a learned linear layer (``trainable=True``),
    optionally with a separate transformation per degree (``per_degree=True``).
    The target number of features is specified by ``num_features``.

    Attributes:
        cutoff (Tensor): Cutoff distance.
        max_angular (int): Maximum spherical harmonic order.
        n_per_l (list): Number of features per degree.

    """

    def __init__(self, alpha = 0, beta = 0, *args, **kwargs):
        """Initialise the Jacobi basis. The default is the Legendre polynomial basis.

        Args:
            alpha (float): real number greater than -1, defining the Jacobi polynomial.
            beta (float): real number greater than -1, defining the Jacobi polynomial.
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

        self.alpha = alpha
        self.beta = beta

    def expand(self, r):
        """Compute the Bernstein polynomial basis.

        Args:
            r (Tensor): Input distances of shape ``[pair]``.

        Returns:
            Expansion of shape ``[pair, num_radial]``.
        """

        r = torch.pi * r.unsqueeze(-1) / self.cutoff 
        cosr = np.array(torch.cos(r))

        y = [
            torch.from_numpy(scipy.special.eval_jacobi(n, self.alpha, self.beta, cosr))
            for n in range(self.num_radial)
        ]

        return y  # -> [pair, num_radial]
