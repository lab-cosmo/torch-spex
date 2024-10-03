from torch.nn import Module

from sphericart.torch import SolidHarmonics as Soh


class SolidHarmonics(Module):
    """Solid harmonics.

    Expands vectors in solid harmonics as defined in ``sphericart``.
    See https://sphericart.readthedocs.io/en/latest/maths.html;
    this computes the Wikipedia definition of real spherical harmonics,
    which are ortho-normalised when integrated over the sphere, times
    ``r**l``.

    Inputs are expected to be a ``[pair, 3]``, i.e. 2D ``Tensor`` of vectors,
    and are returned as a list of tensors, one per degree, with shape ``[pair, 2l+1]``.

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
            Solid harmonics as a list of tensors, one per ``l``,
                of shape ``[pair, 2*l + 1]``.

        """
        # R: [pair, 3 (x,y,z)]
        expansion = self.soh.compute(R)
        return expansion.split(
            self.m_per_l, dim=-1
        )  # -> [[pair, l=0 m=0], [pair, (l=1 m=-1) (l=1 m=0) ...], ...]
