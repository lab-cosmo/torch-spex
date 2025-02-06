import numpy as np

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

    The physical basis has a ``lengthscale`` parameter, which is used to scale the
    inputs before computing the basis, which is defined on the interval ``[0,10]``.
    In the current implementation, this parameter is set to 1.
    """

    def __init__(
        self,
        cutoff,
        **kwargs,
    ):
        """Initializes a physical basis instance.

        Note that ``cutoff<=10`` is required, as the physical basis is only
        defined up to ``r/lengthscale=10``, and currently ``lengthscale=1``.

        See ``TrimmedAndSpline`` for details."""

        if cutoff > 10.0:
            raise ValueError("The physical basis cannot be used for cutoff > 10.0.")

        try:
            from physical_basis import PhysicalBasis

            self.physical_basis = PhysicalBasis()
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "The physical_basis package is required in order to use "
                "PhysicalBasis. Please install spex with the physical extra "
                "(e.g., pip install spex[physical])."
            )

        super().__init__(
            cutoff,
            **kwargs,
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
