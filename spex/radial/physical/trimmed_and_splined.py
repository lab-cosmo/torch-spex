import numpy as np
import torch

import warnings

from .spliner import DynamicSpliner


class TrimmedAndSplined(torch.nn.Module):
    """Base class for trimmed and splined basis functions.

    When implementing a radial expansion, it is often useful to spline the basis
    functions, as this is generally more efficient, even for basis functions that
    can be evaluated analytically.

    Besides a spliner, this class supports the trimming of basis functions based on
    a ranking of the basis functions, here named "eigenvalues". The lower the eigenvalue,
    the more "important" the basis function is. For such bases, often the number of basis
    functions for each angular channel is not constant, but decreases with increasing
    angular momentum ``l``.

    Derived classes must implement the ``compute_eigenvalues`` and ``get_basis_functions``
    methods. The first provides a ranking of basis functions, the second provides their
    functional form.

    Attributes:
        n_per_l (list): Number of basis functions for each angular channel ``l``.
    """

    def __init__(
        self,
        cutoff,
        max_angular=3,
        max_radial=None,
        max_eigenvalue=None,
        n_per_l=None,
        trim=True,
        spliner_accuracy=1e-8,
        normalize=True,
    ):
        """Initialize a trimmed and splined basis.

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
            normalize (bool, optional): Whether to normalize the basis functions,
                measured by squared integral over the interval ``[0, cutoff]``.

        """
        super().__init__()

        # spec
        self.spec = {
            "cutoff": cutoff,
            "max_angular": max_angular,
            "max_radial": max_radial,
            "max_eigenvalue": max_eigenvalue,
            "trim": trim,
            "spliner_accuracy": spliner_accuracy,
            "normalize": normalize,
        }

        # try to preserve original hypers as well as we can
        if n_per_l:
            self.spec["n_per_l"] = n_per_l

        # runtime
        self.per_degree = True
        self.n_per_l = self.get_basis_size(
            cutoff,
            max_radial=max_radial,
            max_angular=max_angular,
            max_eigenvalue=max_eigenvalue,
            n_per_l=n_per_l,
            trim=trim,
        )
        self.max_angular = len(self.n_per_l) - 1
        self.cutoff = cutoff

        R, dR = self.get_spliner_inputs(cutoff, normalize=normalize)

        self.spliner = DynamicSpliner(cutoff, R, dR, accuracy=spliner_accuracy)

    # -- to be implemented in subclass --

    def compute_eigenvalues(self, cutoff, max_l, max_n):
        # This method should return the eigenvalues for the basis functions
        # as a 2D array of shape (max_l, max_n). Currently, it is always
        # called with max_l=50 and max_n=50.
        raise NotImplementedError

    def get_basis_functions(self, cutoff, normalize=True):
        # This should return two functions, R and dR, that compute the basis functions
        # and their derivatives, respectively. The functions should be callable in the
        # form R(x, n, l) and dR(x, n, l) and return tensors.
        raise NotImplementedError

    # -- actual work --

    def forward(self, r):
        """Compute the radial expansion.

        Inputs are expected to be a one-dimensional ``Tensor`` of distances.

        Outputs are always given as a list of two-dimensional ``Tensor`` instances.
        Each element of the list corresponds to one angular channel, in blocks of increasing
        ``l``, and it contains the sample on the first axis and the basis functions for that
        angular channel on the second axis.

        Args:
            r (Tensor): Input distances of shape ``[pair]``.

        Returns:
            Per-degree list expansion of shape ``[pair, self.n_per_l[l]]``.
        """
        # r: [pair]
        basis = self.spliner(r)  # -> [pair, (l=0 n=0) (l=0 n=1) ... (l=1 n=0) ...]

        per_degree = basis.split(self.n_per_l, dim=-1)

        return per_degree

    def get_basis_size(
        self,
        cutoff,
        max_radial=None,
        max_angular=None,
        max_eigenvalue=None,
        n_per_l=None,
        trim=False,
    ):
        # figures out the basis size given the possible combination of hypers
        # basically: (a) we just make a rectangular basis (trim=False) or
        #            (b) we trim based on eigenvalues, setting the threshold based
        #                on respecting max_angular and max_radial (or max_eigenvalue)
        #            (c) we get a premade basis size, in which case we don't do anything.
        #
        #            Note: For compatibility with SphericalExpansion, max_angular will
        #                  always take precedent, if it is not None, n_per_l is guaranteed
        #                  to have max_angular + 1 entries.
        #
        # We return: n_per_l = [number of n for l=0, number of n for l=1, ...]

        if n_per_l is not None:
            if max_angular is not None:
                # max_angular takes precedent, but this may be unexpected
                warnings.warn(
                    "Trimmed radial expansion got n_per_l and max_angular; "
                    "max_angular takes precedent."
                    "To disable this, set max_angular to None."
                )
                return n_per_l[: max_angular + 1]
            else:
                return n_per_l

        if not trim:
            assert max_radial is not None
            assert max_angular is not None
            assert max_eigenvalue is None

            return [max_radial + 1] * (max_angular + 1)

        max_n = 50
        max_l = 50

        eigenvalues_ln = self.compute_eigenvalues(cutoff, max_l, max_n)

        if max_eigenvalue is None:
            if max_angular is None:
                assert max_radial is not None
                assert max_radial <= max_n

                # we go with the max_eigenvalue that leads to max_radial+1 functions at l=0
                max_eigenvalue = eigenvalues_ln[0, max_radial]
                return self.trim_basis(max_eigenvalue, eigenvalues_ln)

            elif max_radial is None:
                assert max_angular is not None
                assert max_angular <= max_l

                # we go with the max_eigenvalue such that there is at least one radial
                # basis function at l=max_angular
                max_eigenvalue = eigenvalues_ln[max_angular, 0]
                return self.trim_basis(max_eigenvalue, eigenvalues_ln)

            else:
                # typical case: max_radial and max_angular are both set
                assert max_radial <= max_n
                assert max_angular <= max_l

                # we first make sure that we have max_radial at l=0
                max_eigenvalue = eigenvalues_ln[0, max_radial]

                # ... then we restrict further
                n_per_l = self.trim_basis(max_eigenvalue, eigenvalues_ln)

                # ... then we make sure that max_angular is respected,
                #     padding as needed
                if len(n_per_l) < (max_angular + 1):
                    return n_per_l + [1] * (max_angular + 1 - len(n_per_l))
                elif len(n_per_l) > (max_angular + 1):
                    return n_per_l[: max_angular + 1]
                else:
                    return n_per_l

        else:
            # special case, not expected to be used with SphericalExpansion
            assert max_radial is None
            assert max_angular is None
            assert max_eigenvalue <= eigenvalues_ln.max()

            return self.trim_basis(max_eigenvalue, eigenvalues_ln)

    def trim_basis(self, max_eigenvalue, eigenvalues_ln):
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

    def get_spliner_inputs(self, cutoff, normalize=False):
        # make functions that accept a batch of distances and return all basis
        # function values (and their derivatives) at once
        # (used only to construct the splines, not for forward pass)

        n_per_l = self.n_per_l
        R, dR = self.get_basis_functions(cutoff, normalize=normalize)

        def values_fn(xs):
            values = []
            for l, number_of_n in enumerate(n_per_l):
                for n in range(number_of_n):
                    values.append(R(xs, n, l))

            return np.stack(values).T

        def derivatives_fn(xs):
            derivatives = []
            for l, number_of_n in enumerate(n_per_l):
                for n in range(number_of_n):
                    derivatives.append(dR(xs, n, l))

            return np.stack(derivatives).T

        return values_fn, derivatives_fn
