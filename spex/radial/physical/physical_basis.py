import numpy as np

from physical_basis import PhysicalBasis as _PhysicalBasis
import torch

from .trimmed_and_splined import TrimmedAndSplined


class PhysicalBasis(TrimmedAndSplined):
    """
    """
    def __init__(
        self,
        cutoff,
        max_radial=10,
        max_angular=None,
        max_eigenvalue=None,
        n_per_l=None,
        trim=True,
        spliner_accuracy=1e-8,
        normalize=True,
    ):
        super().__init__(
            cutoff,
            max_radial,
            max_angular,
            max_eigenvalue,
            n_per_l,
            trim,
            spliner_accuracy,
            normalize,
        )

    def compute_eigenvalues(self, cutoff, max_l, max_n):
        eigenvalues = _PhysicalBasis().E_ln

        assert eigenvalues.shape[0] >= max_l
        assert eigenvalues.shape[1] >= max_n

        return eigenvalues[:max_l, :max_n]


    def get_basis_functions(
        self, cutoff, normalize=True
    ):

        def R(x, n, l):
            device, dtype = x.device, x.dtype
            x = x.numpy()
            ret = _PhysicalBasis().compute(n, l, x)
            if normalize:
                # normalize by square root of sphere volume,
                # excluding sqrt(4pi) which is included in the SH
                ret *= np.sqrt((1 / 3) * cutoff**3)
            ret = torch.tensor(ret, device=device, dtype=dtype)
            return ret

        def dR(x, n, l):
            device, dtype = x.device, x.dtype
            x = x.numpy()
            ret = _PhysicalBasis().compute_derivative(n, l, x)
            if normalize:
                # normalize by square root of sphere volume
                # excluding sqrt(4pi) which is included in the SH
                ret *= np.sqrt((1 / 3) * cutoff**3)
            ret = torch.tensor(ret, device=device, dtype=dtype)
            return ret

        return R, dR
