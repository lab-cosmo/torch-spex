from torch.nn import Module

from sphericart.torch import SphericalHarmonics as Sph

from spex.engine import Specable


class SphericalHarmonics(Module, Specable):
    """Spherical harmonics.

    Expands vectors in spherical harmonics using ``sphericart``.
    See https://sphericart.readthedocs.io/en/latest/maths.html;
    this computes the Wikipedia definition of real spherical harmonics,
    which are ortho-normalised when integrated over the sphere.

    Inputs are expexted to be a ``[pair, 3]``, i.e. 2D ``Tensor`` of vectors,
    and are returned as ``[pair, sum([2*l+1 for i range(max_angular+1)])]``,
    i.e. all spherical harmonics flattened across the second dimension. We use the
    order ``(l=0 m=0) (l=1 m=-1) (l=1 m=0) (l=1 m=+1) (l=2, m=-2) ...``. (Recall
    that each `m` runs from `-l` to `+l` in steps of `1`.)

    The attribute ``m_per_l`` provides the number of ``m`` for each ``l``, and
    can therefore be used in ``torch.split`` to obtain separate features per ``l``,
    at the cost of no longer dealing with a contiguous array.

    """

    def __init__(self, max_angular):
        super().__init__()

        self.max_angular = max_angular

        self.m_per_l = [2 * l + 1 for l in range(max_angular + 1)]

        self.sph = Sph(
            l_max=self.max_angular, normalized=False
        )  # this computes spherical harmonics (not solid harmonics)

        self.spec = {"max_angular": self.max_angular}

    def forward(self, R):
        # R: [pair, 3 (x,y,z)]
        # todo: make special case for "mps" backend (need to move to `cpu`)
        return self.sph.compute(R)  # -> [pair, (l=0 m=0) (l=1 m=-1) (l=1 m=0) ...]
