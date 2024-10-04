import numpy as np

from functools import cache

import scipy as sp
from scipy.special import spherical_jn as j_l

from .trimmed_and_splined import TrimmedAndSplined


class LaplacianEigenstates(TrimmedAndSplined):
    """(Splined) Laplacian Eigenstate Basis.

    Implements the Laplacian eigenstate basis from Bigi et al., doi:10.1063/5.0124363,
    which is composed of spherical Bessel functions and then splined. These functions
    arise as solutions (eigenstates) of the Schrödinger equation with ``H = -∇**2``,
    subject to vanishing boundary conditions at ``|x| = cutoff``.

    This defines radial basis functions that depend on the angular channel ``l``,
    and there is a "basis trimming" feature that reduces the number of basis functions
    for high ``l`` (based on the associated eigenvalue). This is exposed in the hyper-
    parameter selection as ``trim=True``, which then tries to respect ``max_radial``
    and ``max_angular`` to produce an "optimal" basis within those constraints. If
    ``trim=False`` is selected, a "rectangular" basis of size
    ``[max_radial + 1, max_angular + 1]`` is produced. For a further explanation,
    see https://luthaf.fr/rascaline/latest/how-to/le-basis.html.
    """

    def compute_eigenvalues(self, cutoff, max_l, max_n):
        zeros_ln = _compute_zeros(max_l, max_n)
        eigenvalues_ln = zeros_ln**2 / cutoff**2

        return eigenvalues_ln

    def get_basis_functions(self, cutoff, normalize=True):
        # We don't bother with the full equations from doi:10.1063/5.0124363,
        # instead just defining: R_nl(x) ∝ j_l(z_nl x/cutoff) ,
        # where j_l are spherical Bessel functions, and z_nl are their zeroes.
        #
        # We don't deal with the other prefactors because we numerically normalise.
        # This is equivalent to computing N_nl (Eq. A2) and dividing by cutoff**-3/2 ;
        # which you can see by substituting the variables in the integral
        # ∫[0, cutoff] dr r**2 R_nl(r)**2 ) to x = z_nl r / cutoff .

        n_per_l = self.n_per_l
        max_n, max_l = max(n_per_l), len(n_per_l) + 1
        zeros_ln = _compute_zeros(max_l, max_n)

        @cache
        def normalization_factor(n, l):
            if normalize:
                integrand = lambda r: r**2 * j_l(l, zeros_ln[l, n] * r / cutoff) ** 2
                integral, _ = sp.integrate.quad(integrand, 0.0, cutoff)
                return integral ** (-1 / 2)
            else:
                return 1.0

        def R(x, n, l):
            return normalization_factor(n, l) * j_l(l, zeros_ln[l, n] * x / cutoff)

        def dR(x, n, l):
            return (
                normalization_factor(n, l)
                * j_l(l, zeros_ln[l, n] * x / cutoff, derivative=True)  # outer derivative
                * (zeros_ln[l, n] / cutoff)  # inner derivative
            )

        return R, dR


def _compute_zeros(max_angular, max_radial):
    # taken directly from rascaline, who took it from
    # https://scipy-cookbook.readthedocs.io/items/SphericalBesselZeros.html
    # here we "correct" the max_radial/max_angular discrepancy and
    # treat both as inclusive rather than exclusive

    def Jn(r, n):
        return np.sqrt(np.pi / (2 * r)) * sp.special.jv(n + 0.5, r)

    def Jn_zeros(n, nt):
        zeros_j = np.zeros((n + 1, nt), dtype=np.float64)
        zeros_j[0] = np.arange(1, nt + 1) * np.pi
        points = np.arange(1, nt + n + 1) * np.pi
        roots = np.zeros(nt + n, dtype=np.float64)
        for i in range(1, n + 1):
            for j in range(nt + n - i):
                roots[j] = sp.optimize.brentq(Jn, points[j], points[j + 1], (i,))
            points = roots
            zeros_j[i][:nt] = roots[:nt]
        return zeros_j

    return Jn_zeros(max_angular, max_radial + 1)
