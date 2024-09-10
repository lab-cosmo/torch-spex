"""Angular embeddings.

Here, we aim to represent the atom-pair vector connecting two atoms, and in particular
their relative orientation, using spherical harmonics or related functions.

While this is clearly redundant for *one* pair of atoms, where we could just use the
vector itself (or the spherical harmonics with ``l=1``), going to higher order allows
us to reconstruct information about an entire point cloud of neighbours after we have
summed over the entire neighbourhood.

"""

from .solid_harmonics import SolidHarmonics
from .spherical_harmonics import SphericalHarmonics

__all__ = [SphericalHarmonics, SolidHarmonics]
