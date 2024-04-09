"""Species embeddings.

Here, we map an integer, the atomic number or chemical species, to a fixed-size array,
either a one-hot encoding (``Orthogonal``) or a learned embedding (``Alchemical``).

We do this in order to collect information about neighbors into different dimensions
of the output, depending on the species. When we then sum over all neighbours, we are
still able to retain some knowledge about the species of the atoms involved.

"""

from .alchemical import Alchemical
from .orthogonal import Orthogonal

__all__ = [Orthogonal, Alchemical]
