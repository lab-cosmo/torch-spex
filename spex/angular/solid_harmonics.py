from torch.nn import Module

from sphericart.torch import SolidHarmonics as Soh

from spex.engine import Specable


class SolidHarmonics(Module, Specable):
    """Solid harmonics.

    Expands vectors in solid harmonics as defined in ``sphericart``.
    See https://sphericart.readthedocs.io/en/latest/maths.html;
    this computes the Wikipedia definition of real spherical harmonics,
    which are ortho-normalised when integrated over the sphere, times
    ``r**l``.

    Inputs are expexted to be a ``[pair, 3]``, i.e. 2D ``Tensor`` of vectors,
    and are returned as ``[pair, sum([2*l+1 for i in range(max_angular+1)])]``,
    i.e. all solid harmonics flattened across the second dimension. We use the
    order ``(l=0 m=0) (l=1 m=-1) (l=1 m=0) (l=1 m=+1) (l=2, m=-2) ...``. (Recall
    that each ``m`` runs from ``-l`` to ``+l`` in steps of ``1``.)

    The attribute ``m_per_l`` provides the number of ``m`` for each ``l``, and
    can therefore be used in ``torch.split`` to obtain separate features per ``l``.

    Attributes:
        max_angular (int): The maximum solid harmonic order ``l`` to compute.
        m_per_l (list): Number of ``m`` for each ``l``.

    """

    def __init__(self, max_angular):
        """Initialise SolidHarmonics.

        Args:
            max_angular (int): The maximum solid harmonic order ``l`` to compute.
        """
        super().__init__()

        self.max_angular = max_angular

        self.m_per_l = [2 * l + 1 for l in range(max_angular + 1)]

        self.soh = Soh(l_max=self.max_angular)

        self.spec = {"max_angular": self.max_angular}

    def forward(self, R):
        """Compute solid harmonics for input vectors.

        Args:
            R (Tensor): Input vectors of shape ``[pair, 3]``.

        Returns:
            Solid harmonics of shape
                ``[pair, sum([2*l+1 for i in range(max_angular+1)])]``.

        """
        # R: [pair, 3 (x,y,z)]
        # todo: make special case for "mps" backend (need to move to `cpu`)
        return self.soh.compute(R)  # -> [pair, (l=0 m=0) (l=1 m=-1) (l=1 m=0) ...]
