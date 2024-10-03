import numpy as np

from physical_basis import PhysicalBasis as _PhysicalBasis

from .trimmed_and_splined import TrimmedAndSplined


class PhysicalBasis(TrimmedAndSplined):
    """
    """

    def __init__(self, cutoff, n_per_l=None, normalize=True, spliner_accuracy=1e-8):
        super().__init__(
            cutoff,
            n_per_l=n_per_l,
            normalize=normalize,
            spliner_accuracy=spliner_accuracy,
        )
        self.physical_basis = _PhysicalBasis()

    def compute_eigenvalues(self, cutoff, max_l, max_n):
        eigenvalues = self.physical_basis.E_ln

        assert eigenvalues.shape[0] >= max_l
        assert eigenvalues.shape[1] >= max_n

        return eigenvalues[:max_l, :max_n]


    def get_basis_functions(
        self, cutoff, normalize=True
    ):

        def R(x, n, l):
            ret = self.physical_basis.compute(n, l, x)
            if normalize:
                # normalize by square root of sphere volume,
                # excluding sqrt(4pi) which is included in the SH
                ret *= np.sqrt((1 / 3) * cutoff**3)
            return ret

        def dR(x, n, l):
            ret = self.physical_basis.compute_derivative(n, l, x)
            if normalize:
                # normalize by square root of sphere volume
                # excluding sqrt(4pi) which is included in the SH
                ret *= np.sqrt((1 / 3) * cutoff**3)
            return ret

        return R, dR
