import numpy as np
import torch

from physical_basis import PhysicalBasis as _PhysicalBasis

from .trimmed_and_splined import TrimmedAndSplined


class PhysicalBasis(TrimmedAndSplined):
    """
    (Splined) Physical Radial Basis.

    Implements a "physical" radial basis, which is a radial basis that can be
    obtained by solving a physically motivated differential equation. This is
    implemented in the ``physical_basis`` package which this class wraps and
    depends on.

    This defines radial basis functions that depend on the angular channel ``l``,
    and there is a "basis trimming" feature that reduces the number of basis functions
    for high ``l`` (based on the associated eigenvalue). This is exposed in the hyper-
    parameter selection as ``trim=True``, which then tries to respect ``max_radial``
    and ``max_angular`` to produce an "optimal" basis within those constraints. If
    ``trim=False`` is selected, a "rectangular" basis of size
    ``[max_radial + 1, max_angular + 1]`` is produced.

    This class inherits most of its functionality from ``TrimmedAndSplined``.
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
        """
        Compute the eigenvalues for the physical basis,
        as required by the ``TrimmedAndSplined`` class.

        Args:
            cutoff (float): Cutoff radius.
            max_l (int): Maximum angular channel.
            max_n (int): Maximum radial channel.

        Returns:
            np.ndarray: Eigenvalues for the physical basis as a 2D array
                with shape ``(max_l, max_n)``.
        """

        eigenvalues = _PhysicalBasis().E_ln

        assert eigenvalues.shape[0] >= max_l
        assert eigenvalues.shape[1] >= max_n

        return eigenvalues[:max_l, :max_n]


    def get_basis_functions(
        self, cutoff, normalize=True
    ):
        """
        Returns functions that compute the radial basis functions and their derivatives
        for the Laplacian eigenstates basis.
        """
        # The normalization here is geometric. It assumes that the basis functions
        # vanish (or almost vanish) at the cutoff radius.

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
