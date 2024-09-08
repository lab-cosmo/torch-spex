import numpy as np
import torch

from functools import cache

import scipy as sp
from scipy.special import spherical_jn as j_l

from spex.engine import Specable

from .cutoff import get_cutoff_function
from .spliner import DynamicSpliner

from physical_basis import PhysicalBasis as _PhysicalBasis


class PhysicalBasis(torch.nn.Module, Specable):
    """Physical (Radial, Splined, Cut-off) Basis.

    Implements the "physical" basis.

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
        max_radial=10,
        max_angular=None,
        max_eigenvalue=None,
        n_per_l=None,
        trim=True,
        spliner_accuracy=1e-8,
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

    R, dR = get_basis_functions(cutoff, normalize=normalize)

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

    physical_basis = _PhysicalBasis()
    eigenvalues_ln = physical_basis.E_ln

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
    normalize=True,
):
    physical_basis = _PhysicalBasis()

    def R(x, n, l):
        ret = physical_basis.compute(n, l, x)
        if normalize:
            # normalize by square root of sphere volume, excluding sqrt(4pi) which is included in the SH
            ret *= np.sqrt((1 / 3) * cutoff**3)
        return ret

    def dR(x, n, l):
        ret = physical_basis.compute_derivative(n, l, x)
        if normalize:
            # normalize by square root of sphere volume, excluding sqrt(4pi) which is included in the SH
            ret *= np.sqrt((1 / 3) * cutoff**3)
        return ret

    return R, dR
