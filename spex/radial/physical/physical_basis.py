import numpy as np
import torch

try:
    from physical_basis import PhysicalBasis as _PhysicalBasis
except ImportError:
    raise ImportError(
        "The `physical_basis` package is required in order to use the "
        "physical basis. Please install spex with the `physical` extra "
        "(e.g., `pip install spex[physical]`)."
    )

from .trimmed_and_splined import TrimmedAndSplined


class PhysicalBasis(TrimmedAndSplined):
    """(Splined) Physical Radial Basis.

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

    The physical basis has a lengthscale parameter which represents the lengthscale
    of the short-range interactions in the physical system. In the current
    implementation, this parameter is set to 1 (in the units provided by the user).
    """

    def __init__(
        self,
        cutoff,
        max_radial=None,
        max_angular=None,
        max_eigenvalue=None,
        n_per_l=None,
        trim=True,
        spliner_accuracy=1e-8,
        normalize=True,
    ):
        """
        Initializes a physical basis instance.

        All arguments are passed to the base class, ``TrimmedAndSplined``.
        Cutoffs larger than 10 are not supported at the moment.
        """
        # this needs to be initialized before super().__init__(), as
        # it is used in the two derived functions that are called by it
        self.physical_basis = _PhysicalBasis()

        if cutoff > 10.0:
            # the ``physical_basis`` package does not support
            # r/lengthscale > 10.0
            raise ValueError("The physical basis cannot be used for cutoff > 10.0.")

        super().__init__(
            cutoff=cutoff,
            max_radial=max_radial,
            max_angular=max_angular,
            max_eigenvalue=max_eigenvalue,
            n_per_l=n_per_l,
            trim=trim,
            spliner_accuracy=spliner_accuracy,
            normalize=normalize,
        )

    def compute_eigenvalues(self, cutoff, max_l, max_n):
        eigenvalues = self.physical_basis.E_ln

        assert eigenvalues.shape[0] >= max_l
        assert eigenvalues.shape[1] >= max_n

        return eigenvalues[:max_l, :max_n]

    def get_basis_functions(self, cutoff, normalize=True):
        # The normalization here is geometric. It assumes that the basis functions
        # vanish (or almost vanish) at the cutoff radius.

        def R(x, n, l):
            device, dtype = x.device, x.dtype
            x = x.numpy()
            ret = self.physical_basis.compute(n, l, x)
            if normalize:
                # normalize by square root of sphere volume,
                # excluding sqrt(4pi) which is included in the SH
                ret *= np.sqrt((1 / 3) * cutoff**3)
            ret = torch.tensor(ret, device=device, dtype=dtype)
            return ret

        def dR(x, n, l):
            device, dtype = x.device, x.dtype
            x = x.numpy()
            ret = self.physical_basis.compute_derivative(n, l, x)
            if normalize:
                # normalize by square root of sphere volume
                # excluding sqrt(4pi) which is included in the SH
                ret *= np.sqrt((1 / 3) * cutoff**3)
            ret = torch.tensor(ret, device=device, dtype=dtype)
            return ret

        return R, dR
