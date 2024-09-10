import numpy as np
import torch

from functools import cache

import scipy as sp
from scipy.special import spherical_jn as j_l

from .cutoff import get_cutoff_function
from .spliner import DynamicSpliner


class LaplacianEigenstates(torch.nn.Module):
    """Laplacian Eigenstate (Radial, Splined, Cut-off) Basis.

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

    We modify the basis functions with a cutoff function ensuring to ensure a smooth
    decay to zero at ``cutoff``, using a ``"shifted_cosine"``.

    A discontinuous cutoff, ``"step"``, is recommended for testing only.

    Inputs are expected to be a one-dimensional ``Tensor`` of distances.

    Outputs are always given in a two-dimensional ``Tensor``; all basis functions are
    flattened across one dimension and arranged in blocks of increasing ``l``. The
    ``n_per_l`` attribute of instances of this class provides the sizes required for
    ``torch.split`` to obtain features per ``l``.

    Attributes:
        n_per_l (list): Number of basis functions for each angular channel ``l``.

    """

    def __init__(
        self,
        cutoff,
        max_radial=20,
        max_angular=None,
        max_eigenvalue=None,
        n_per_l=None,
        trim=True,
        spliner_accuracy=1e-2,
        cutoff_function="shifted_cosine",
        normalize=True,
    ):
        """Initialise the Laplacian Eigenstate basis.

        If ``trim=False``, nothing complicated happens and there is a constant number of
        basis functions for each ``l``. The basis is therefore "rectangular" in shape:
        For each of the ``max_angular + 1`` choices of  ``l`` there are ``max_radial + 1``
        basis functions.

        If ``trim=True``, the basis is trimmed to become smaller with increasing ``l``,
        based on the associated eigenvalues. We do our best to respect ``max_radial``,
        ``max_angular``, and ``max_eigenvalue`` choices (see ``get_basis_size``).

        If ``n_per_l`` is provided, everything else is ignored and this is used to define
        the basis size.

        Args:
            cutoff (float): Cutoff radius.
            max_radial (int, optional): Number of radial basis functions.
            max_angular (int, optional): Number of angular channels to consider.
            max_eigenvalue (float, optional): Maximum eigenvalue to be used for trimming.
            n_per_l (list, optional): Number of basis functions for each angular channel.
                (Overrides other options for basis selection.)
            trim (bool, optional): Whether to trim the basis.
            spliner_accuracy (float, optional): The accuracy of the spliner.
            cutoff_function (str, optional): The cutoff function to use, either
                ``shifted_cosine`` or ``step``.
            normalize (bool, optional): Whether to normalize the basis functions,
                measured by squared integral over the interval ``[0, cutoff]``.

        """
        super().__init__()

        # spec
        self.spec = {
            "cutoff": cutoff,
            "max_radial": max_radial,
            "max_angular": max_angular,
            "max_eigenvalue": max_eigenvalue,
            "trim": trim,
            "spliner_accuracy": spliner_accuracy,
            "cutoff_function": cutoff_function,
            "normalize": normalize,
        }

        # try to preserve original hypers as well as we can
        if n_per_l:
            self.spec["n_per_l"] = n_per_l

        # runtime
        self.n_per_l = get_basis_size(
            cutoff,
            max_radial=max_radial,
            max_angular=max_angular,
            max_eigenvalue=max_eigenvalue,
            n_per_l=n_per_l,
            trim=trim,
        )
        self.max_angular = len(self.n_per_l) - 1

        R, dR = get_spliner_inputs(cutoff, self.n_per_l, normalize=normalize)

        self.spliner = DynamicSpliner(cutoff, R, dR, accuracy=spliner_accuracy)

        self.cutoff_fn = get_cutoff_function(cutoff, cutoff_function)

    def forward(self, r):
        """Compute the radial expansion.

        Args:
            r (Tensor): Input distances of shape ``[pair]``.

        Returns:
            Expansion of shape ``[pair, sum([n for n in n_per_l])]``.
        """
        # r: [pair]
        cutoff = self.cutoff_fn(r)  # -> [pair]
        basis = self.spliner(r)  # -> [pair, (l=0 n=0) (l=0 n=1) ... (l=1 n=0) ...]

        return cutoff.unsqueeze(1) * basis  # -> [pair, (l=0 n=0) (l=0 n=1) ...]


def get_spliner_inputs(cutoff, n_per_l, normalize=False):
    # make functions that accept a batch of distances and return all basis
    # function values (and their derivatives) at once
    # (used only to construct the splines, not for forward pass)

    R, dR = get_basis_functions(cutoff, n_per_l, normalize=normalize)

    def values_fn(xs):
        values = []
        for l, number_of_n in enumerate(n_per_l):
            for n in range(number_of_n):
                values.append(R(xs, n, l))

        return torch.stack(values).T

    def derivatives_fn(xs):
        derivatives = []
        for l, number_of_n in enumerate(n_per_l):
            for n in range(number_of_n):
                derivatives.append(dR(xs, n, l))

        return torch.stack(derivatives).T

    return values_fn, derivatives_fn


# -- hypers --


def get_basis_size(
    cutoff,
    max_radial=None,
    max_angular=None,
    max_eigenvalue=None,
    n_per_l=None,
    trim=False,
):
    # figures out the basis size given the possible combination of hypers
    # basically: (a) we just make a rectangular basis (trime=False) or
    #            (b) we trim based on eigenvalues (which in turn can be set by choosing
    #                a maximum basis size)
    #            (c) we get a premade basis size, in which case we don't do anything.
    # We return: n_per_l = [number of n for l=0, number of n for l=1, ...]

    if n_per_l is not None:
        # work was already done
        return n_per_l

    if not trim:
        assert max_radial is not None
        assert max_angular is not None
        assert max_eigenvalue is None

        return [max_radial + 1] * (max_angular + 1)

    # we'll be trimming by eigenvalue, so we calculate all eigenvalues up to some maximum
    max_l = 100
    max_n = 100

    zeros_ln = compute_zeros(max_l, max_n)
    eigenvalues_ln = zeros_ln**2 / cutoff**2

    if max_eigenvalue is None:
        if max_angular is None:
            assert max_radial is not None
            assert max_radial <= max_n

            # we go with the max_eigenvalue that leads to max_radial+1 functions at l=0
            max_eigenvalue = eigenvalues_ln[0, max_radial]
            return trim_basis(max_eigenvalue, eigenvalues_ln)

        elif max_radial is None:
            assert max_angular is not None
            assert max_angular <= max_l

            # we go with the max_eigenvalue such that there is at least one radial
            # basis function at l=max_angular
            max_eigenvalue = eigenvalues_ln[max_angular, 0]
            return trim_basis(max_eigenvalue, eigenvalues_ln)

        else:
            assert max_radial <= max_n
            assert max_angular <= max_l

            # we first make sure that the max_radial at l=0 is within bounds,
            max_eigenvalue = eigenvalues_ln[0, max_radial]

            # ... then we restrict further
            n_per_l = trim_basis(max_eigenvalue, eigenvalues_ln)
            return n_per_l[:max_angular]

    else:
        assert max_radial is None
        assert max_angular is None
        assert max_eigenvalue <= eigenvalues_ln.max()

        return trim_basis(max_eigenvalue, eigenvalues_ln)

    raise ValueError("invalid combination of hypers")


def trim_basis(max_eigenvalue, eigenvalues_ln):
    # retain only basis functions with eigenvalue <= max_eigenvalue
    n_per_l = []
    for ell in range(eigenvalues_ln.shape[0]):
        n_radial = len(np.where(eigenvalues_ln[ell] <= max_eigenvalue)[0])
        n_per_l.append(n_radial)
        if n_per_l[-1] == 0:
            # all eigenvalues for this l are over the threshold
            n_per_l.pop()
            break

    return n_per_l


# -- actual basis functions --


def get_basis_functions(
    cutoff,
    n_per_l,
    normalize=True,
):
    # We don't bother with the full equations from doi:10.1063/5.0124363,
    # instead just defining: R_nl(x) ∝ j_l(z_nl x/cutoff) ,
    # where j_l are spherical Bessel functions, and z_nl are their zeroes.
    #
    # We don't deal with the other prefactors because we numerically normalise.
    # This is equivalent to computing N_nl (Eq. A2) and dividing by cutoff**-3/2 ;
    # which you can see by substituting the variables in the integral
    # ∫[0, cutoff] dr r**2 R_nl(r)**2 ) to x = z_nl r / cutoff .

    max_n, max_l = max(n_per_l), len(n_per_l) + 1
    zeros_ln = compute_zeros(max_l, max_n)

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


# -- numerics --


def compute_zeros(max_angular: int, max_radial: int) -> np.ndarray:
    # taken directly from rascaline, who took it from
    # https://scipy-cookbook.readthedocs.io/items/SphericalBesselZeros.html
    # here we "correct" the max_radial/max_angular discrepancy and
    # treat both as inclusive rather than exclusive

    def Jn(r: float, n: int) -> float:
        return np.sqrt(np.pi / (2 * r)) * sp.special.jv(n + 0.5, r)

    def Jn_zeros(n: int, nt: int) -> np.ndarray:
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
